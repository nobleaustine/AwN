import ast
import yaml

class ClassInfoExtractor(ast.NodeVisitor):

    #info: dict of infos of class
    def __init__(self):
        self.info = {}

    def visit_ClassDef(self, node):

        # infos needed
        class_name = node.name
        base_class_name = " "
        for base in node.bases:
            if isinstance(base, ast.Attribute):
              base_class_name = base.attr 
             
        class_name = f"{class_name}({base_class_name})"
        constructor_arguments = {}
        function_arguments = {}
        class_attributes = []
        class_methods = {}
        
        # loop to go to each node and continue if it is a function
        for func in node.body:
            function_arguments = {}
            if isinstance(func, ast.FunctionDef):
                # for constructor
                if func.name == '__init__':

                    arguments1 = reversed([arg.arg for arg in func.args.args if arg.arg != "self"])
                    arguments2 = reversed([arg.arg for arg in func.args.args if arg.arg != "self"])
                    defaults =reversed([defa.value for defa in func.args.defaults])
                   
                    for arg,dft in zip(arguments1,defaults) :
                        constructor_arguments[arg]=dft
                    
                    for arg in arguments2:
                        if arg not in constructor_arguments.keys():
                            constructor_arguments[arg]= None
                        
                    # getting assignments in constructor by checking self 
                    for statement in func.body:
                        if isinstance(statement, ast.Assign):
                            for target in statement.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    class_attributes.append(target.attr)
                # avoiding inbuilt functions
                elif not func.name.startswith('__'):
                    
                    function_name = func.name
                    # getting passed arguments
                    arguments1 = reversed([arg.arg for arg in func.args.args if arg.arg != "self"])
                    arguments2 = reversed([arg.arg for arg in func.args.args if arg.arg != "self"])
                    defaults =reversed([defa.value for defa in func.args.defaults])
                   
                    for arg,dft in zip(arguments1,defaults) :
                        function_arguments[arg]=dft
                    
                    for arg in arguments2:
                        if arg not in function_arguments.keys():
                            function_arguments[arg]= None
                    
                    return_variables = []
                    # getting return values
                    for statement in func.body:
                        if isinstance(statement, ast.Return):
                            if isinstance(statement.value, ast.Name):
                                return_variables.append(statement.value.id)
                            else:
                                return_variables.append("__")
                    # adding class arguments and return_variables
                    class_methods[function_name] = {
                        "arguments": function_arguments,
                        "returns": return_variables
                    }

        self.info[class_name] = {

            "constructor_arguments": constructor_arguments,
            "class_attributes": class_attributes,
            "class_methods": class_methods,
            
        }

def extract_class_info(file_path):

    # functions = {}
    # f_arguments = {}
    # f_returns = []
    with open(file_path, "r") as file:
        source_code = file.read()

    tree = ast.parse(source_code)
    # for node in ast.walk(tree):
    #     if isinstance(node, ast.FunctionDef) and node.args.args[0].arg !="self":
    #         arguments1 = reversed([arg.arg for arg in node.args.args ])
    #         arguments2 = reversed([arg.arg for arg in node.args.args ])
    #         defaults =reversed([defa.value for defa in node.args.defaults])
            
    #         for arg,dft in zip(arguments1,defaults) :
    #             f_arguments[arg]=dft
            
    #         for arg in arguments2:
    #             if arg not in f_arguments.keys():
    #                 f_arguments[arg]= None

    #         for statement in node.body:
    #                     if isinstance(statement, ast.Return):
                            
    #                         if isinstance(statement.value, ast.Name):
    #                             f_returns.append(statement.value.id)
    #                         else:
    #                             f_returns.append("__")

    #         functions[node.name] = {
    #                     "arguments": f_arguments,
    #                     "return_variables": f_returns
    #                 }
    
    extractor = ClassInfoExtractor()
    extractor.visit(tree)
    # extractor.info.update(functions)
    return extractor.info

def get_yaml(file_path,yaml_path):

    info = extract_class_info(file_path)

    with open(yaml_path, "w") as yaml_file:
        yaml.dump(info, yaml_file, default_flow_style=False,sort_keys=False)



