from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from prompt_templates import (
    Separators_template,
    Nest_template,
    Tag_template,
    Result_example_template,
    Problem_template,
    Function_gen_template,
    Code_gen_template,
    Python_code_template
)

# model_4_mini = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.8, top_p=0.95)
# model_4o = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.8, top_p=0.95)

def setup_chain(prompt_template, model, parser):
        prompt = PromptTemplate.from_template(prompt_template)
        return prompt | model | parser

def constrained_self_and_plan_chains(model_name):
    Separators_chain = setup_chain(Separators_template, model_name, StrOutputParser())
    Nest_chain = setup_chain(Nest_template, model_name, StrOutputParser())
    Tag_chain = setup_chain(Tag_template, model_name, StrOutputParser())
    Result_example_chain = setup_chain(Result_example_template, model_name, StrOutputParser())

    function_gen_chain = setup_chain(Function_gen_template, model_name, StrOutputParser())
    code_gen_chain = setup_chain(Code_gen_template, model_name, StrOutputParser())
    python_code_chain = setup_chain(Python_code_template, model_name, StrOutputParser())

    @chain
    def custom_chain(data_piece):
        separators = Separators_chain.invoke({'data': data_piece})    
        nest = Nest_chain.invoke({'data': data_piece})
        tags = Tag_chain.invoke({'data': data_piece})
        result_json = Result_example_chain.invoke({
            'data': data_piece,
            'separators': separators,
            'nest': nest,
            'tags': tags
        })

        Problem = PromptTemplate.from_template(Problem_template).format(
            data=data_piece,
            json_data=result_json,
            separators=separators,
            nest=nest,
            tags=tags
        )
        
        function_gen = function_gen_chain.invoke({'requirement': Problem})
        process_code = code_gen_chain.invoke({'solving_process': function_gen}) 
        parseCode = python_code_chain.invoke({'parseCode': process_code})
       
        # return function_gen, parseCode, result_json, separators, nest, tags
        return function_gen, parseCode, result_json, separators, nest, tags
    
    return custom_chain


