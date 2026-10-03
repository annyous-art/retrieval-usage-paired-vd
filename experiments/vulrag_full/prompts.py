"""Verbatim official prompt functions; see vendor/LICENSE and provenance.json."""

def generate_extract_prompt(CVE_id, CVE_description, modified_lines, code_before_change, code_after_change):
    prefix_str = f"""This is a code snippet with a vulnerability {CVE_id}:
'''
{code_before_change}
'''
The vulnerability is described as follows:
{CVE_description}
"""

    # extract purpose prompt
    purpose_prompt = f"""{prefix_str}
What is the purpose of the function in the above code snippet? \
Please summarize the answer in one sentence with following format: Function purpose: \"\"
"""

    # extract function prompt
    function_prompt = f"""{prefix_str}
Please summarize the functions of the above code snippet in the list format without other \
explanation: \"The functions of the code snippet are: 1. 2. 3.\"
"""

    # extract analysis prompt
    analysis_prompt = f"""{prefix_str}
The correct way to fix it is by adding/deleting
'''
{modified_lines}
'''
."""

    if modified_lines["added"] != []:
        analysis_prompt += f"""The code after modification is as follows:\n'''\n{code_after_change}\n'''\n"""

    analysis_prompt += """Why is the above modification necessary?"""

    knowledge_extraction_prompt = """
I want you to act as a vulnerability detection expert and organize vulnerability knowledge based on the above \
vulnerability repair information. Please summarize the generalizable specific behavior of the code that \
leads to the vulnerability and the specific solution to fix it. Format your findings in JSON.
Here are some examples to guide you on the level of detail expected in your extraction:
Example 1:
{
    "vulnerability_behavior": {
        'vulnerability_cause_description': 'Lack of proper handling for asynchronous events during device removal process.',
        'trigger_condition': 'A physically proximate attacker unplugs a device while the removal function is executing, \
leading to a race condition and use-after-free vulnerability.',
        'specific_code_behavior_causing_vulnerability': 'The code does not cancel pending work associated with a specific \
functionality before proceeding with further cleanup during device removal. This can result in a use-after-free scenario if \
the device is unplugged at a critical moment.'
    }, 
    'solution': 'To mitigate the vulnerability, it is necessary to cancel any pending work related to the specific \
functionality before proceeding with further cleanup during device removal. This ensures that the code handles asynchronous \
events properly and prevents the use-after-free vulnerability. In this case, the solution involves adding a line to cancel the \
pending work associated with the specific functionality before continuing with the cleanup process.'
}
Note that in the 'solution' field of your response's JSON, the solution should be described in natural language format. Do not nest dictionaries or arrays within the 'solution' field. Plus, do not nest within other field either. Your answer should be exactly the same format as the example we provide.
Please be mindful to omit specific resource names in your descriptions to ensure the knowledge remains generalized. \
For example, instead of writing mutex_lock(&dmxdev->mutex), simply use mutex_lock.
"""

    return purpose_prompt, function_prompt, analysis_prompt, knowledge_extraction_prompt

def generate_extraction_prompt_for_vulrag(code_snippet):
    prefix_str = f"""This is a code snippet: \n{code_snippet}\n"""
    # extract purpose prompt
    purpose_prompt = prefix_str + (
        "What is the purpose of the function in the above code snippet? "
        "Please summarize the answer in one sentence with the following format: "
        "Function purpose: \"\""
    )

    # extract function prompt
    function_prompt = prefix_str + (
        "Please summarize the functions of the above code snippet "
        "in the list format without other explanation: "
        "\"The functions of the code snippet are: 1. 2. 3.\""
    )

    return purpose_prompt, function_prompt

def generate_detect_vul_prompt_with_response_in_HTML(code_snippet, cve_knowledge) -> str:
    return f"""I want you to act as a vulnerability detection expert, given the following code snippet and related vulnerability knowledge, please detect whether there is a similar vulnerability in the code snippet.
Code Snippet:
'''
{code_snippet}
'''
Vulnerability Knowledge:
In a similar code scenario, the following vulnerabilities have been found:
'''
{cve_knowledge}
'''
Please check if the above code snippet contains similar vulnerability behaviors mentioned in the vulnerability knowledge. Perform a step-by-step analysis and conclude your response with either <result> YES </result> or <result> NO </result>.
"""

def generate_detect_sol_prompt_with_response_in_HTML(code_snippet, cve_knowledge) -> str:
    return f"""I want you to act as a vulnerability detection expert, given the following code snippet and related vulnerability knowledge, please detect whether there are similar necessary solution behaviors in the code snippet, which can prevent the occurrence of related vulnerabilities in the vulnerability knowledge.
Code Snippet:
'''
{code_snippet}
'''
Vulnerability Knowledge:
In a similar code scenario, the following vulnerabilities have been found:
'''
{cve_knowledge}
'''
Please check if the above code snippet contains similar solution behaviors mentioned in the vulnerability knowledge. Perform a step-by-step analysis and conclude your response with either <result> YES </result> or <result> NO </result>.
"""
