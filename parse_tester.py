"""
Independent Parse & Verification Tester

This module allows testing parse code and verification independently 
without affecting the main watermark system. Implements fault tolerance
by isolating errors at the verification level.

Usage:
    python parse_tester.py --raw-log path/to/log.txt --schema schema.json --mode verify
"""

import sys
import os
import json
import argparse
import logging
import pandas as pd
import traceback
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from verification_enhanced import (
    EnhancedVerificationModule,
    create_default_schema,
    validate_separator_consistency,
    ErrorCollector,
    suggest_repairs_via_llm,
    build_pydantic_schema_from_rules,
    export_jsonschema,
    VerificationStatus
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParseTester:
    """
    Independent tester for parse code and verification pipeline.
    Implements fault tolerance by isolating verification errors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            Configuration for verification (tolerances, etc.)
        """
        self.config = config or self._default_config()
        self.logger = logger
        self.verification_history = []
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default verification configuration"""
        return {
            "local_tolerance": 0.1,
            "global_tolerance": 0.05,
            "signature_threshold": 0.9,
            "max_repair_iterations": 2,
            "enable_repair": True,
            "strict_validation": False
        }
    
    def test_separator_consistency(
        self,
        raw_text: str,
        parsed_df: pd.DataFrame,
        separators: list = None,
        context: str = "log"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Test separator consistency independently.
        
        Parameters
        ----------
        raw_text : str
            Raw log/text before parsing
        parsed_df : pd.DataFrame
            Parsed result
        separators : list
            Separators to check (default: auto-detect)
        context : str
            Context hint for validation
        
        Returns
        -------
        Tuple[bool, Dict]
            - passed: Whether test passed
            - report: Detailed test report
        """
        self.logger.info("=" * 60)
        self.logger.info("TEST: Separator Consistency")
        self.logger.info("=" * 60)
        
        try:
            passed, report = validate_separator_consistency(
                raw_text=raw_text,
                parsed_df=parsed_df,
                separators=separators,
                context=context
            )
            
            self.logger.info(f"✓ Separator test: {'PASS' if passed else 'FAIL'}")
            self.logger.info(f"Report: {json.dumps(report, indent=2, default=str)}")
            
            return passed, report
        
        except Exception as e:
            self.logger.error(f"✗ Separator test failed with exception: {e}")
            self.logger.error(traceback.format_exc())
            return False, {"error": str(e), "exception": type(e).__name__}
    
    def test_schema_validation(
        self,
        data_sample: pd.DataFrame,
        custom_schema: Optional[Dict] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Test schema validation independently.
        
        Parameters
        ----------
        data_sample : pd.DataFrame
            Sample data to validate
        custom_schema : Optional[Dict]
            Custom schema (auto-generate if None)
        
        Returns
        -------
        Tuple[bool, Dict]
            - passed: Whether validation passed
            - report: Detailed validation report
        """
        self.logger.info("=" * 60)
        self.logger.info("TEST: Schema Validation")
        self.logger.info("=" * 60)
        
        try:
            # Generate or use provided schema
            schema = custom_schema or create_default_schema(data_sample)
            self.logger.info(f"Schema: {json.dumps(schema, indent=2, default=str)}")
            
            # Create verifier
            verifier = EnhancedVerificationModule(
                schema=schema,
                **self.config
            )
            
            # Validate
            validation_engine = verifier.validation_engine
            is_valid, cleaned_df, invalid_rows = validation_engine.validate_dataframe(data_sample)
            
            report = {
                "passed": is_valid,
                "total_rows": len(data_sample),
                "valid_rows": len(cleaned_df),
                "invalid_rows": len(invalid_rows),
                "invalid_row_details": invalid_rows[:5]  # Limit to first 5
            }
            
            self.logger.info(f"✓ Schema validation: {'PASS' if is_valid else 'FAIL'}")
            self.logger.info(f"Report: {json.dumps(report, indent=2, default=str)}")
            
            return is_valid, report
        
        except Exception as e:
            self.logger.error(f"✗ Schema validation failed: {e}")
            self.logger.error(traceback.format_exc())
            return False, {"error": str(e), "exception": type(e).__name__}
    
    def test_error_collection(
        self,
        data_sample: pd.DataFrame,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test error collection and LLM suggestions.
        
        Parameters
        ----------
        data_sample : pd.DataFrame
            Sample data with potential errors
        schema : Dict
            Schema for validation
        
        Returns
        -------
        Dict[str, Any]
            Error collection and suggestions report
        """
        self.logger.info("=" * 60)
        self.logger.info("TEST: Error Collection & LLM Suggestions")
        self.logger.info("=" * 60)
        
        try:
            # Validate and collect errors
            verifier = EnhancedVerificationModule(schema=schema, **self.config)
            is_valid, cleaned_df, invalid_rows = verifier.validation_engine.validate_dataframe(data_sample)
            
            if not invalid_rows:
                self.logger.info("No errors found in data")
                return {"errors": [], "suggestions": []}
            
            # Collect errors
            error_collector = ErrorCollector()
            for row_idx, errors in invalid_rows:
                for err in errors:
                    error_collector.add_error(
                        error_type=err.get("type", "unknown"),
                        message=err.get("message", ""),
                        severity=err.get("severity", "warning"),
                        field=err.get("field"),
                        value=err.get("input_value")
                    )
            
            error_summary = error_collector.get_summary()
            self.logger.info(f"Collected {error_summary['total_errors']} errors")
            self.logger.info(f"Error categories: {error_summary['categories']}")
            
            # Get repair suggestions
            suggestions = suggest_repairs_via_llm(
                errors=error_summary,
                schema=schema,
                llm_chain=None,
                use_placeholder=True
            )
            
            report = {
                "errors_collected": error_summary["total_errors"],
                "categories": error_summary["categories"],
                "suggestions": suggestions.get("repairs", []),
                "source": suggestions.get("source", "")
            }
            
            self.logger.info(f"Generated {len(report['suggestions'])} repair suggestions")
            self.logger.info(f"Report: {json.dumps(report, indent=2, default=str)}")
            
            return report
        
        except Exception as e:
            self.logger.error(f"✗ Error collection test failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e), "exception": type(e).__name__}
    
    def test_full_verification(
        self,
        original_df: pd.DataFrame,
        watermarked_df: pd.DataFrame,
        extracted_signature: list,
        expected_signature: list,
        raw_text: Optional[str] = None,
        separator: Optional[str] = None,
        custom_schema: Optional[Dict] = None
    ) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
        """
        Run full verification pipeline with fault tolerance.
        
        Parameters
        ----------
        original_df : pd.DataFrame
            Original data before watermarking
        watermarked_df : pd.DataFrame
            Data after watermarking
        extracted_signature : list
            Extracted watermark bits
        expected_signature : list
            Expected watermark bits
        raw_text : Optional[str]
            Raw text for separator check
        separator : Optional[str]
            Separator token
        custom_schema : Optional[Dict]
            Custom schema (auto-generate if None)
        
        Returns
        -------
        Tuple[str, pd.DataFrame, Dict]
            - status: Verification status (SUCCESS/REPAIRED/FAILED)
            - verified_df: Verified dataset
            - report: Detailed report
        """
        self.logger.info("=" * 60)
        self.logger.info("TEST: Full Verification Pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Generate schema
            schema = custom_schema or create_default_schema(original_df)
            
            # Create verifier
            verifier = EnhancedVerificationModule(
                schema=schema,
                **self.config
            )
            
            # Run verification with fault isolation
            status, verified_df, report = verifier.verify_dataset(
                original_df=original_df,
                watermarked_df=watermarked_df,
                extracted_signature=extracted_signature,
                expected_signature=expected_signature,
                raw_text=raw_text,
                separator=separator
            )
            
            self.verification_history.append({
                "status": status.value if hasattr(status, "value") else str(status),
                "timestamp": pd.Timestamp.now(),
                "report": report
            })
            
            self.logger.info(f"✓ Verification completed: {status}")
            self.logger.info(f"Detailed report:\n{json.dumps(report, indent=2, default=str)}")
            
            return str(status), verified_df, report
        
        except Exception as e:
            self.logger.error(f"✗ Verification pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return graceful fallback (fault tolerance)
            fallback_report = {
                "error": str(e),
                "exception": type(e).__name__,
                "fallback": True,
                "recommendation": "Use original watermarked data with caution"
            }
            
            return "FAILED", watermarked_df, fallback_report
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate summary of all tests run"""
        return {
            "total_tests": len(self.verification_history),
            "tests": self.verification_history
        }


def main():
    """CLI interface for parse testing"""
    parser = argparse.ArgumentParser(
        description="Independent Parse & Verification Tester"
    )
    parser.add_argument(
        "--mode",
        choices=["separator", "schema", "errors", "full"],
        default="full",
        help="Test mode to run"
    )
    parser.add_argument(
        "--original-data",
        help="Path to original CSV data"
    )
    parser.add_argument(
        "--watermarked-data",
        help="Path to watermarked CSV data"
    )
    parser.add_argument(
        "--raw-text",
        help="Path to raw log/text file"
    )
    parser.add_argument(
        "--separator",
        default="\n",
        help="Separator token to check (default: newline)"
    )
    parser.add_argument(
        "--schema",
        help="Path to custom schema JSON"
    )
    parser.add_argument(
        "--expected-sig",
        default="1,0,1,0",
        help="Comma-separated expected signature bits"
    )
    parser.add_argument(
        "--extracted-sig",
        default="1,0,1,0",
        help="Comma-separated extracted signature bits"
    )
    parser.add_argument(
        "--output",
        help="Output file for test report"
    )
    
    args = parser.parse_args()
    
    logger.info("Parse Tester Starting")
    tester = ParseTester()
    
    # Load custom schema if provided
    custom_schema = None
    if args.schema:
        with open(args.schema, 'r') as f:
            custom_schema = json.load(f)
    
    # Run appropriate test
    if args.mode == "full":
        if not all([args.original_data, args.watermarked_data]):
            print("Error: --original-data and --watermarked-data required for full mode")
            sys.exit(1)
        
        original_df = pd.read_csv(args.original_data)
        watermarked_df = pd.read_csv(args.watermarked_data)
        
        raw_text = None
        if args.raw_text:
            with open(args.raw_text, 'r') as f:
                raw_text = f.read()
        
        extracted_sig = [int(x.strip()) for x in args.extracted_sig.split(',')]
        expected_sig = [int(x.strip()) for x in args.expected_sig.split(',')]
        
        status, verified_df, report = tester.test_full_verification(
            original_df=original_df,
            watermarked_df=watermarked_df,
            extracted_signature=extracted_sig,
            expected_signature=expected_sig,
            raw_text=raw_text,
            separator=args.separator,
            custom_schema=custom_schema
        )
    
    elif args.mode == "separator":
        if not all([args.raw_text, args.watermarked_data]):
            print("Error: --raw-text and --watermarked-data required for separator mode")
            sys.exit(1)
        
        with open(args.raw_text, 'r') as f:
            raw_text = f.read()
        
        watermarked_df = pd.read_csv(args.watermarked_data)
        tester.test_separator_consistency(raw_text, watermarked_df, [args.separator])
    
    elif args.mode == "schema":
        if not args.watermarked_data:
            print("Error: --watermarked-data required for schema mode")
            sys.exit(1)
        
        watermarked_df = pd.read_csv(args.watermarked_data)
        tester.test_schema_validation(watermarked_df, custom_schema)
    
    elif args.mode == "errors":
        if not args.watermarked_data:
            print("Error: --watermarked-data required for errors mode")
            sys.exit(1)
        
        watermarked_df = pd.read_csv(args.watermarked_data)
        schema = custom_schema or create_default_schema(watermarked_df)
        tester.test_error_collection(watermarked_df, schema)
    
    # Save report if requested
    if args.output:
        report = tester.generate_test_report()
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
