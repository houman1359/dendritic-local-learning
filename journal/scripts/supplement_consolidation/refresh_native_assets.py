"""Record current native SI renders without rewriting the frozen input registry."""
from pathlib import Path
import hashlib,json
import pymupdf as fitz

J=Path(__file__).resolve().parents[2]
RENDERS={
 'N9':('input_coverage_mnist',1),
 'N7':('mnist_dictionary_geometry',3),
 'N21':('local_gate_controls',4),
 'N22':('physical_architecture',7),
 'N31':('measured_transfer_geometry',4),
}
def main():
 records={}
 for key,(name,count) in RENDERS.items():
  path=J/'figures/supplementary'/f'figure_{name}_native.pdf'
  with fitz.open(path) as doc:
   metadata=json.loads(doc.metadata['keywords'])
   declared=metadata.get('panels',[])
   letters=[]
   for block in doc[0].get_text('dict')['blocks']:
    for line in block.get('lines',[]):
     for span in line['spans']:
      if span['text'] in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count]) and abs(span['size']-9)<.05 and 'Bold' in span['font']:
       letters.append(dict(letter=span['text'],bbox=span['bbox'],origin=span['origin']))
   assert sorted(x['letter'] for x in letters)==list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count]),(key,letters)
   records[key]=dict(path=str(path.relative_to(J)),old_label='fig:si_'+name,letters=sorted(letters,key=lambda x:x['letter']),
      builder=f'scripts/build_supplementary_figure_{name}_native.py',
      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      note='Current editorial render; the frozen original registry remains unchanged.')
 (J/'configs/supplement_consolidation/current_native_assets.json').write_text(json.dumps(records,indent=2)+'\n')
 print('Recorded',len(records),'current native SI assets.')
if __name__=='__main__':main()
