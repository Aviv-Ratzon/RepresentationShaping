import json
import re

def resolve_conflicts():
    # Read the original file content
    with open('analyze_linear_network_original.ipynb', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Resolve execution count conflicts - keep the higher number
    content = re.sub(r'<<<<<<< HEAD\s+"execution_count": (\d+),=======\s+"execution_count": (\d+),>>>>>>> [a-f0-9]+', 
                     lambda m: f'"execution_count": {max(int(m.group(1)), int(m.group(2)))},', content)
    
    # Resolve learning rate conflicts - keep the newer value (1)
    content = re.sub(r'<<<<<<< HEAD\s+C\.learning_rate = 0\.01\nC\.L=5\n=======\s+C\.learning_rate = 1\nC\.L=0\n>>>>>>> [a-f0-9]+', 
                     'C.learning_rate = 1\nC.L=0', content)
    
    # Resolve max_move conflicts - keep the newer value (25)
    content = re.sub(r'<<<<<<< HEAD\s+C\.max_move = 5\nC\.hidden_size = 21 # \(C\.length_corridors\[0\]\+2\*C\.max_move\+1 \+ 1\)\*len\(C\.length_corridors\)\nC\.num_epochs = 10000\n=======\s+C\.max_move = 25\n', 
                     'C.max_move = 25\n', content)
    
    # Resolve hidden_size conflicts - keep the newer value
    content = re.sub(r'C\.hidden_size = 21 # \(C\.length_corridors\[0\]\+2\*C\.max_move\+1 \+ 1\)\*len\(C\.length_corridors\)\nC\.num_epochs = 10000\n=======\s+C\.max_move = 25\nC\.hidden_size = 50\*3-1 # \(C\.length_corridors\[0\]\+2\*C\.max_move\+1 \+ 1\)\*len\(C\.length_corridors\)\nC\.num_epochs = 1000000\n>>>>>>> [a-f0-9]+', 
                     'C.hidden_size = 50*3-1 # (C.length_corridors[0]+2*C.max_move+1 + 1)*len(C.length_corridors)\nC.num_epochs = 1000000', content)
    
    # Remove all remaining conflict markers
    content = re.sub(r'<<<<<<< HEAD.*?=======.*?>>>>>>> [a-f0-9]+\n?', '', content, flags=re.DOTALL)
    
    # Write the resolved content
    with open('analyze_linear_network.ipynb', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    resolve_conflicts()
