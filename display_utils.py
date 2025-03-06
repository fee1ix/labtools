from IPython.display import display,clear_output, HTML

def set_display_for_dataframe():
    css = """
<style type='text/css'>
.dataframe th, .dataframe td {
    white-space: nowrap;
    text-align: left;
    padding: 4px;
    line-height: 1; /* This sets the line height to one */
}
</style>
    """
    display(HTML(css))
    return 


import math
def print_prompt_messages(prompt_messages):
    for message in prompt_messages:
        print(message['role'])
        print(message['content'], end='\n\n')


def print_multicol(cols):
    col_width=math.floor(100/len(cols))
    html_content=""
    for col in cols:
        html_content+=f"""
<div style="float: left; width: {col_width}%;">
    <pre style="white-space: pre-wrap;">{col}</pre>
</div>
"""
    display(HTML(html_content))
