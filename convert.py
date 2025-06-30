import nbformat
from nbconvert import PythonExporter

def convert_notebook_to_python(notebook_path, output_path=None):
    # Read the notebook
    with open("C:\projects\Speech-Emotion-Recognition-with-MFCC-main\Speech Emotion Recogntion.ipynb", 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Create Python exporter
    exporter = PythonExporter()
    
    # Convert to Python
    (body, resources) = exporter.from_notebook_node(notebook)
    
    # Determine output path
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '.py')
    
    # Write the Python file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(body)
    
    print(f"Converted {notebook_path} to {output_path}")

# Usage
convert_notebook_to_python('your_notebook.ipynb')
