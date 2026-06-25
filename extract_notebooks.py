import json
import os
import glob

notebooks = glob.glob('c:/Users/User/delivery-time-prediction/notebooks/*.ipynb')
with open('c:/Users/User/delivery-time-prediction/notebook_summaries.txt', 'w', encoding='utf-8') as outfile:
    for nb_path in notebooks:
        filename = os.path.basename(nb_path)
        outfile.write(f'\n============================\n--- {filename} ---\n============================\n')
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
        except Exception as e:
            outfile.write(f'Error loading json: {e}\n')
            continue
        
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell.get('source', []))
                if len(content.strip()) > 0:
                    outfile.write(f'MD: {content[:300].replace(chr(10), " ")}\n')
            elif cell['cell_type'] == 'code':
                source = ''.join(cell.get('source', []))
                source_lower = source.lower()
                # we want to catch the best params, best score, etc.
                if 'score' in source_lower or 'best' in source_lower or 'r2' in source_lower or 'mae' in source_lower or 'rmse' in source_lower or 'accuracy' in source_lower or 'model' in source_lower:
                    outputs = cell.get('outputs', [])
                    out_texts = []
                    for out in outputs:
                        if out.get('output_type') == 'stream':
                            out_texts.append(''.join(out.get('text', [])))
                        elif out.get('output_type') == 'execute_result':
                            data = out.get('data', {})
                            if 'text/plain' in data:
                                out_texts.append(''.join(data['text/plain']))
                    if out_texts:
                        combined_out = ''.join(out_texts)
                        outfile.write(f'CODE OUT (from "{source[:50].replace(chr(10), " ")}"): {combined_out[:500].replace(chr(10), " ")}\n')
