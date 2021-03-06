from typing import Dict

class Report:
    def __init__(self, variables: Dict, template: str = 'default_template.md'):
        self.variables = variables
        self.template = template

        self.load_template()
        
    def load_template(self):
        with open(self.template, 'r') as f:
            self.content = f.read()

        return self.content

    def process_content(self):
        content = self.content

        # TODO - check if all loaded variables are in the report template.
        # TODO - check if there is some variable that exists in report but weren't loaded.

        for k,v in self.variables.items():
            # content = self.content.replace(f"${k.lower()}", v)
            content = content.replace(f"${k.upper()}", str(v))
        
        return content

    def to_md(self, filename: str):
        self.content = self.process_content()

        with open(filename, 'w') as f:
            f.write(self.content)
