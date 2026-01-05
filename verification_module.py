"""
Verification Module for WatermarkHub
----------------------------------

This module implements the verification, repair and retry logic for the
WatermarkHub pipeline.  It is inspired by the validation and repair
framework described in recent literature on structured LLM outputs.  The
goal of the module is to ensure that watermarked datasets produced by
the injection stage are structurally valid, respect local and global
constraints, and contain the correct watermark signature.  When issues
are detected, the module attempts to repair the dataset automatically.
If repair is not possible, a re‑ask mechanism can be invoked to
generate a new output from the underlying LLM with explicit error
feedback.  This implementation is self contained and does not depend
on any external API (e.g. OpenAI).  To integrate this module into
WatermarkHub, import and instantiate the ``VerificationModule`` class
and call its ``verify_dataset`` method after the injection stage.

The module makes use of ``pydantic`` for declarative schema
definitions and validation.  It also provides simple helpers for
automatic repair and signature verification.  The design follows the
guidelines laid out in the paper “WatermarkHub” (see Section 4 of the
paper, where the verification module is described as verifying,
repairing and retrying【686692174918128†L600-L609】).  It also adopts
the clear structure definition, automatic validation and re‑ask
mechanism advocated by the structured output framework【825192367772108†L140-L176】.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Import adaptive density calculation (avoid circular dependency: only needed at runtime)
try:
    from watermark import compute_adaptive_gamma  # type: ignore
except Exception:  # Tolerant handling, use fallback logic if not found
    def compute_adaptive_gamma(cfg, num_rows: int, num_index_attrs: int) -> int:  # type: ignore
        return getattr(cfg, 'GAMMA', 100)
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, create_model


class VerificationError(Exception):
    """Custom error type for verification failures."""
    pass


def generate_model_from_schema(schema: Dict[str, Dict[str, Any]]) -> BaseModel:
    """
    Dynamically create a Pydantic model from a simple schema definition.

    Parameters
    ----------
    schema: Dict[str, Dict[str, Any]]
        A dictionary describing the expected fields.  Each key is the
        field name and the value is a dictionary with keys ``type`` and
        optionally ``default``.  The ``type`` value should be a Python
        type (e.g. ``str`` or ``int``).

    Returns
    -------
    BaseModel
        A Pydantic model class that can be used to validate objects.

    Examples
    --------
    >>> schema = {
    ...     "name": {"type": str},
    ...     "age": {"type": int, "default": 0},
    ... }
    >>> Model = generate_model_from_schema(schema)
    >>> Model(name="Alice", age=30)
    Model(name='Alice', age=30)
    """
    fields: Dict[str, Tuple[Any, Field]] = {}
    for field_name, spec in schema.items():
        field_type = spec.get("type", Any)
        default = spec.get("default", ...)
        fields[field_name] = (field_type, Field(default=default))
    return create_model("DynamicRecord", **fields)  # type: ignore[misc]


class ValidationEngine:
    """
    Component responsible for validating data against a given schema.

    This class encapsulates schema definition and validation logic.  It
    can validate individual records (dictionaries) or entire pandas
    DataFrames.  When validation fails it returns a list of errors
    rather than raising an exception, allowing the caller to decide
    whether to repair the data or trigger a re‑ask.
    """

    def __init__(self, schema: Dict[str, Dict[str, Any]]) -> None:
        self.schema_dict = schema
        self.model = generate_model_from_schema(schema)

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[BaseModel], Optional[List[str]]]:
        """
        Validate a single record against the schema.

        Returns a tuple ``(is_valid, instance, errors)``.  If ``is_valid``
        is True, ``instance`` is the parsed Pydantic object; otherwise
        ``errors`` contains human readable error messages.
        """
        try:
            instance = self.model(**record)
            return True, instance, None
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = ".".join([str(x) for x in err.get("loc", [])])
                msg = err.get("msg", "invalid")
                errors.append(f"Field '{loc}': {msg}")
            return False, None, errors

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, List[Tuple[int, List[str]]]]:
        """
        Validate an entire DataFrame against the schema.

        Returns
        -------
        is_valid : bool
            True if all records are valid.
        cleaned_df : pd.DataFrame
            DataFrame with only valid rows.
        invalid_rows : List[Tuple[int, List[str]]]
            A list of tuples where each tuple contains the index of the
            invalid row and the list of error messages for that row.
        """
        valid_rows = []
        invalid_rows: List[Tuple[int, List[str]]] = []
        for idx, row in df.iterrows():
            record = row.to_dict()
            ok, _, errors = self.validate_record(record)
            if ok:
                valid_rows.append(row)
            else:
                assert errors is not None
                invalid_rows.append((idx, errors))
        if invalid_rows:
            cleaned_df = pd.DataFrame(valid_rows).reset_index(drop=True)
            return False, cleaned_df, invalid_rows
        else:
            return True, df.reset_index(drop=True), []


class RepairEngine:
    """
    Component responsible for repairing records that fail validation.

    The repair strategy is simplistic: missing fields are filled using
    default values defined in the schema; fields with incorrect types
    are cast when possible.  For numeric fields the value is clipped
    into an allowed range if provided.
    """

    def __init__(self, schema: Dict[str, Dict[str, Any]]) -> None:
        self.schema = schema

    def repair_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to repair a record according to the schema.

        Parameters
        ----------
        record: Dict[str, Any]
            The record to repair.

        Returns
        -------
        Dict[str, Any]
            The repaired record.  If a field cannot be repaired, its
            original value is preserved.
        """
        repaired = record.copy()
        for field_name, spec in self.schema.items():
            # Fill missing fields with defaults
            if field_name not in repaired or repaired[field_name] is None:
                if "default" in spec:
                    repaired[field_name] = spec["default"]
            # Attempt type casting
            expected_type = spec.get("type")
            if expected_type and field_name in repaired and repaired[field_name] is not None:
                value = repaired[field_name]
                # Simple type cast if possible
                try:
                    if not isinstance(value, expected_type):
                        repaired[field_name] = expected_type(value)  # type: ignore[misc]
                except Exception:
                    # leave original value if cast fails
                    pass
            # Clip numeric values to an allowed range if provided
            if field_name in repaired and "min" in spec and "max" in spec:
                try:
                    num_val = float(repaired[field_name])
                    num_val = max(spec["min"], min(spec["max"], num_val))
                    # cast back to original type
                    repaired[field_name] = expected_type(num_val)  # type: ignore[misc]
                except Exception:
                    # ignore if casting fails
                    pass
        return repaired

    def repair_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to repair an entire DataFrame.

        Returns
        -------
        pd.DataFrame
            The repaired DataFrame.
        """
        repaired_rows = []
        for _, row in df.iterrows():
            repaired_rows.append(self.repair_record(row.to_dict()))
        return pd.DataFrame(repaired_rows)


def verify_watermark_signature(
    extracted_bits: List[int], expected_bits: List[int], threshold: float = 0.9
) -> bool:
    """
    Compare extracted watermark bits to the expected signature.

    This function computes the proportion of matching bits and returns
    True if it exceeds the specified threshold.  The threshold allows
    for majority voting correction as described in the paper【686692174918128†L600-L609】.

    Parameters
    ----------
    extracted_bits: List[int]
        The bits extracted from the watermarked dataset.
    expected_bits: List[int]
        The reference watermark bits (signature).
    threshold: float, optional
        The minimum fraction of bits that must match for the signature
        to be considered valid.  Defaults to 0.9.

    Returns
    -------
    bool
        True if the signature is considered valid, False otherwise.
    """
    if len(extracted_bits) != len(expected_bits):
        return False
    correct = sum(1 for a, b in zip(extracted_bits, expected_bits) if a == b)
    return correct / len(expected_bits) >= threshold


class VerificationModule:
    """
    High level component encapsulating verification logic.

    The module coordinates validation, repair and signature checking.  It
    operates on pandas DataFrames and interacts with WatermarkHub’s
    injection and detection functions by accepting both the original
    data and the watermarked data.  It also supports a simple re‑ask
    mechanism to request a new output when verification fails.
    """

    def __init__(
        self,
        schema: Dict[str, Dict[str, Any]],
        local_tolerance: float = 0.1,
        global_tolerance: float = 0.05,
        signature_threshold: float = 0.9,
        max_retries: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        schema: dict
            Schema definition for the expected data format.
        local_tolerance: float
            Maximum allowed relative change for any numeric field (local
            constraint).  For example, 0.1 means each modified numeric
            value should not deviate more than ±10% from the original.
        global_tolerance: float
            Maximum allowed fraction of modified records (global
            constraint).  For example, 0.05 means no more than 5% of
            records should be altered.
        signature_threshold: float
            Threshold for watermark signature matching.
        max_retries: int
            Maximum number of re‑ask attempts when verification fails.
        """
        self.validation_engine = ValidationEngine(schema)
        self.repair_engine = RepairEngine(schema)
        self.local_tolerance = local_tolerance
        self.global_tolerance = global_tolerance
        self.signature_threshold = signature_threshold
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_local_constraints(
        self,
        original_df: pd.DataFrame,
        watermarked_df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
    ) -> bool:
        """
        Verify local usability constraints: each numeric field in the
        watermarked data should deviate from the original by no more
        than ``local_tolerance`` fraction of the original absolute value.

        Parameters
        ----------
        original_df: pd.DataFrame
            The original unmodified data.
        watermarked_df: pd.DataFrame
            The watermarked data.
        numeric_cols: List[str], optional
            List of numeric columns to check.  If None, all columns of
            numeric dtype will be considered.

        Returns
        -------
        bool
            True if the constraint holds for all checked values.
        """
        if numeric_cols is None:
            numeric_cols = [c for c in original_df.columns if pd.api.types.is_numeric_dtype(original_df[c])]
        for col in numeric_cols:
            orig = original_df[col].astype(float)
            wm = watermarked_df[col].astype(float)
            # compute relative deviation, add small epsilon to avoid division by zero
            eps = 1e-8
            deviations = (wm - orig).abs() / (orig.abs() + eps)
            if (deviations > self.local_tolerance).any():
                self.logger.debug(
                    "Local constraint violated in column '%s': max deviation %.4f exceeds tolerance %.4f",
                    col,
                    deviations.max(),
                    self.local_tolerance,
                )
                return False
        return True

    def check_global_constraints(
        self,
        original_df: pd.DataFrame,
        watermarked_df: pd.DataFrame,
        columns_to_check: Optional[List[str]] = None,
    ) -> bool:
        """
        Verify the global usability constraint: only a small fraction of
        records should be modified (up to ``global_tolerance`` of the
        dataset size).

        Parameters
        ----------
        original_df: pd.DataFrame
            The original unmodified data.
        watermarked_df: pd.DataFrame
            The watermarked data.
        columns_to_check: List[str], optional
            Columns to compare for modifications.  If None, all columns
            are compared.

        Returns
        -------
        bool
            True if the fraction of modified rows does not exceed the
            global tolerance.
        """
        if columns_to_check is None:
            columns_to_check = list(original_df.columns)
        # Determine which rows differ across the selected columns
        diffs = (original_df[columns_to_check] != watermarked_df[columns_to_check]).any(axis=1)
        fraction_modified = diffs.mean() if len(original_df) > 0 else 0.0
        if fraction_modified > self.global_tolerance:
            self.logger.debug(
                "Global constraint violated: %.2f%% of rows modified, exceeds tolerance %.2f%%",
                fraction_modified * 100,
                self.global_tolerance * 100,
            )
            return False
        return True

    def verify_dataset(
        self,
        original_df: pd.DataFrame,
        watermarked_df: pd.DataFrame,
        extracted_signature: List[int],
        expected_signature: List[int],
        cfg: Optional[Any] = None,
        index_attributes: Optional[List[str]] = None,
    ) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """
        Verify a watermarked dataset and attempt repairs when needed.

        Parameters
        ----------
        original_df: pd.DataFrame
            Original dataset before watermarking.
        watermarked_df: pd.DataFrame
            Dataset after watermark injection.
        extracted_signature: List[int]
            Watermark bits extracted from ``watermarked_df`` (e.g. via
            the detection function).
        expected_signature: List[int]
            The expected watermark signature bits.

        Returns
        -------
        Tuple[bool, pd.DataFrame, Dict[str, Any]]
            ``is_valid``: Whether the final dataset passes all checks.
            ``final_df``: The verified (and possibly repaired) dataset.
            ``report``: A dictionary containing details about
            validation, repairs and signature matching.
        """
        # Calculate and record adaptive gamma (if cfg and index_attributes are provided)
        if cfg is not None:
            num_rows = len(original_df)
            num_index_attrs = len(index_attributes) if index_attributes else 1
            adaptive_gamma = compute_adaptive_gamma(cfg, num_rows, num_index_attrs)
            # Estimate expected modification rate based on adaptive_gamma: p_row_modified ≈ 1 - (1 - 1/gamma)^{num_index_attrs}
            if adaptive_gamma > 0:
                expected_fraction = 1 - (1 - 1/float(adaptive_gamma)) ** max(1, num_index_attrs)
            else:
                expected_fraction = 1.0
            scale = getattr(cfg, 'TOLERANCE_SCALING', 1.8)
            tol_min = getattr(cfg, 'GLOBAL_TOLERANCE_MIN', 0.01)
            tol_max = getattr(cfg, 'GLOBAL_TOLERANCE_MAX', 0.25)
            adjusted_global_tolerance = max(tol_min, min(tol_max, expected_fraction * scale))
        else:
            adaptive_gamma = None
            adjusted_global_tolerance = None
            expected_fraction = None

        report: Dict[str, Any] = {
            "structure_valid": None,
            "local_constraints": None,
            "global_constraints": None,
            "signature_valid": None,
            "retries": 0,
            "invalid_rows": None,
            "adaptive_gamma": adaptive_gamma,
            "num_index_attributes": len(index_attributes) if index_attributes else None,
            "adjusted_global_tolerance": adjusted_global_tolerance,
            "expected_modification_fraction": expected_fraction if cfg is not None else None,
        }
        current_df = watermarked_df.copy().reset_index(drop=True)
        # Validate and repair up to max_retries
        for attempt in range(self.max_retries + 1):
            report["retries"] = attempt
            # Structural validation
            structure_ok, cleaned_df, invalid_rows = self.validation_engine.validate_dataframe(current_df)
            report["structure_valid"] = structure_ok
            report["invalid_rows"] = invalid_rows
            if not structure_ok:
                # repair invalid records
                repaired_df = self.repair_engine.repair_dataframe(current_df)
                # keep only rows that were originally present to preserve ordering
                current_df = repaired_df.reset_index(drop=True)
                self.logger.debug(
                    "Attempt %d: repaired %d invalid rows", attempt, len(invalid_rows)
                )
                continue
            # Check constraints
            local_ok = self.check_local_constraints(original_df, current_df)
            # If adaptive value available, use runtime-adjusted global tolerance for checking (without permanently modifying instance property)
            if adjusted_global_tolerance is not None:
                original_global_tol = self.global_tolerance
                self.global_tolerance = adjusted_global_tolerance
                global_ok = self.check_global_constraints(original_df, current_df)
                self.global_tolerance = original_global_tol
                report["used_global_tolerance"] = adjusted_global_tolerance
            else:
                global_ok = self.check_global_constraints(original_df, current_df)
            report["local_constraints"] = local_ok
            report["global_constraints"] = global_ok
            if not (local_ok and global_ok):
                # repair by resetting offending rows to original values
                repaired_df = current_df.copy()
                diffs = (original_df != current_df).any(axis=1)
                # for local violations: we clip values; for global violations: revert changes beyond tolerance
                if not local_ok:
                    # clip numeric columns using repair engine
                    numeric_cols = [c for c in original_df.columns if pd.api.types.is_numeric_dtype(original_df[c])]
                    for col in numeric_cols:
                        orig = original_df[col].astype(float)
                        wm = current_df[col].astype(float)
                        eps = 1e-8
                        deviations = (wm - orig).abs() / (orig.abs() + eps)
                        mask = deviations > self.local_tolerance
                        repaired_df.loc[mask, col] = orig.loc[mask]
                if not global_ok:
                    modified_mask = diffs
                    # revert changes of rows beyond allowed percentage
                    # compute allowed number of modified rows
                    allowed = int(len(original_df) * self.global_tolerance)
                    modified_indices = list(modified_mask[modified_mask].index)
                    if len(modified_indices) > allowed:
                        # revert the excess modified rows to original values
                        to_revert = modified_indices[allowed:]
                        repaired_df.loc[to_revert, :] = original_df.loc[to_revert, :]
                current_df = repaired_df.reset_index(drop=True)
                continue
            # Check signature
            signature_ok = verify_watermark_signature(
                extracted_signature, expected_signature, self.signature_threshold
            )
            report["signature_valid"] = signature_ok
            if signature_ok:
                # All checks passed
                return True, current_df, report
            else:
                # In this simplified implementation, if signature fails we cannot repair
                self.logger.debug(
                    "Attempt %d: signature mismatch, extracted bits %s != expected bits %s",
                    attempt,
                    extracted_signature,
                    expected_signature,
                )
                # Break out since we can't repair watermark bits
                break
        # If we reach here, verification failed
        return False, current_df, report

    def reask(self, *args: Any, **kwargs: Any) -> None:
        """
        Placeholder for a re‑ask mechanism.

        In a full implementation this method would call the underlying
        LLM with explicit feedback to regenerate outputs when the
        verification and repair steps fail【825192367772108†L170-L176】.  Here we
        provide a stub so that downstream systems can override it.
        """
        raise NotImplementedError(
            "Re‑ask mechanism is not implemented in this example."
        )
