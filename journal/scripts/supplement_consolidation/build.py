#!/usr/bin/env python3
"""Reflow selected native PDF panels without rasterizing or changing data.

All input identities, panel crops, placements and old provenance records are
written to configs/supplement_consolidation/manifest.json. Original sheets
remain immutable inputs.

2026-09-10 (SI_PLAN v2 section 5):
  * one paste scale per figure, capped at ``PASTE_SCALE`` = 1.0, so the source
    type ladder is the printed type ladder.  A figure whose frozen panel
    inventory cannot be laid out at 1.0 inside ``HEIGHT_CAP`` is pasted at one
    uniform smaller scale and is recorded in the exemption list written to
    ``configs/supplement_consolidation/audit_report.json``; the within-figure
    ladder stays uniform either way, which is the defect section 4d names.
  * new lettering is Nimbus Sans Bold at ``journal_style.PT_LETTER``; restored
    axis labels and native legends use ``PT_BASE``/``PT_EMPH`` and ``LW_DATA``.
  * ``configs/supplement_consolidation/reference_map.json`` is generated here,
    with a per-panel ``omitted_panel_status`` record for every dropped panel.
  * every curated PDF is audited by ``figure_canvas.audit_native_pdf``.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re,sys,math
from pathlib import Path
import fitz
import numpy as np
from matplotlib import font_manager

J=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(J/'scripts'))
from specification import (FIGURES, REMOVED_SHARED_REGIONS, SHARED_LEGENDS, WHOLE_CROPS,
 PANEL_BOUNDS, PANEL_REDACTIONS, PANEL_PATCHES, PANEL_TEXT, NATIVE_LEGENDS,
 EXPLICIT_NUMERICAL_SOURCES, NUMERICAL_SOURCE_OVERRIDES, EXTRA_ASSETS, CAPTION_APPEND, ALIAS_EXTRA,
 ALIAS_BLACKLIST, PANEL_CONTENT, PANEL_REASONS, SOURCE_DATA_DIRS,
 ASSET_PATH_OVERRIDES, SCALE_EXEMPTIONS, CONSOLIDATED_FIGURE_NUMBERS)
from tex_sources import expanded_tex
import journal_style as JS
OUT=J/'figures/supplementary/curated'; CFG=J/'configs/supplement_consolidation'; TEX=J/'supplementary/curated'
REG=json.loads((HERE/'original_assets.json').read_text())
W=518.4
PASTE_SCALE=1.0        # the source ladder is the printed ladder
MIN_SCALE=0.74         # floor for a frozen inventory that cannot fit at 1.0
HEIGHT_CAP=540.        # the graphics standard's cap for one supplement sheet
MARGIN=4.; GUTTER=8.; GUTTER3=2.; LETTER_BAND=15.
LETTER_GUTTER=11.0  # 9-pt bold letter advance plus the 3.4-pt content gap
# Preserve the reviewed scientific grouping; trimming blank crop margins must
# not silently move a task or comparison into a different row.
ROW_COUNTS={
 'credit_validation':(2,1), 'reliability_gain':(2,2),
 'same_span_conditioning':(2,2), 'image_generalization':(3,2),
 'mechanistic_chain':(3,), 'input_coverage_depth':(3,1), 'branch_conflict_controls':(2,2),
 'ancestry_coefficients':(1,2), 'physical_optimizer':(2,1),
 'anatomy_capacity_controls':(2,2), 'inhibitory_spatial_controls':(1,2,2),
}


def column_widths(rows):
 """Common decorated-content widths for repeated row grids."""
 widths={}
 for sub in rows:
  for col,q in enumerate(sub):
   widths[len(sub),col]=max(widths.get((len(sub),col),0),q['width'])
 return widths


def prepared_panel(key, oldletter):
 """Remove only declared lettering/masks; retain the scientific vector art."""
 source=fitz.open(J/REG[key]['path']);sp=source[0];masks=[]
 for letter in REG[key]['letters']:
  sp.add_redact_annot(fitz.Rect(letter['bbox']),fill=None)
 for rr in REMOVED_SHARED_REGIONS.get(key,[])+PANEL_REDACTIONS.get((key,oldletter),[]):
  sp.add_redact_annot(fitz.Rect(rr),fill=None);masks.append(fitz.Rect(rr))
 if REG[key]['letters'] or masks:sp.apply_redactions(images=0,graphics=0,text=0)
 for rr in masks:sp.draw_rect(rr,color=None,fill=(1,1,1),width=0)
 for patch in PANEL_PATCHES.get((key,oldletter),[]):
  with fitz.open(J/REG[patch['source']]['path']) as aux:
   sp.show_pdf_page(fitz.Rect(patch['target_bbox']),aux,0,clip=fitz.Rect(patch['bbox']))
 return source


def trimmed_panel_bounds(key, oldletter, bbox):
 """Measure empty crop margins, without rasterizing the publication output.

 The source crop remains the ownership boundary. Only genuinely blank pixels
 are trimmed, after old panel letters are removed. Explicit restored text is
 included so it can never be cropped off by this layout-only operation.
 """
 with prepared_panel(key,oldletter) as doc:
  clip=fitz.Rect(bbox);zoom=4.0
  pix=doc[0].get_pixmap(matrix=fitz.Matrix(zoom,zoom),clip=clip,
                       colorspace=fitz.csGRAY,alpha=False)
  arr=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width)
  yy,xx=np.where(arr<250)
  if not len(xx):return clip
  tight=fitz.Rect((pix.x+xx.min())/zoom,(pix.y+yy.min())/zoom,
                  (pix.x+xx.max()+1)/zoom,(pix.y+yy.max()+1)/zoom)
  # Keep the original crop wherever restored axis text will be inserted.
  if PANEL_TEXT.get((key,oldletter)):return clip
  return fitz.Rect(tight.x0-.6,tight.y0-.6,tight.x1+.6,tight.y1+.6)&clip

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spans(page):return [s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
def union(rects):
 out=fitz.Rect(rects[0])
 for r in rects[1:]:out|=r
 return out

for _k,_p in ASSET_PATH_OVERRIDES.items():REG[_k]['path']=_p

# Panels drawn for the supplement itself (SI_PLAN M1, AMENDMENTS B3).
for _k,_v in EXTRA_ASSETS.items():
 _rec=dict(_v)
 if not (J/_rec['path']).exists():
  raise FileNotFoundError(f"{_rec['path']} is missing; run scripts/build_si_restored_panels.py")
 _rec['sha256']=sha(J/_rec['path']);REG[_k]=_rec

def make_bounds(key):
 rec=REG[key];doc=fitz.open(J/rec['path']); p=doc[0]; letters=rec['letters']
 for rr in REMOVED_SHARED_REGIONS.get(key,[]):p.add_redact_annot(rr,fill=(1,1,1))
 if key in REMOVED_SHARED_REGIONS:p.apply_redactions(images=0,graphics=0,text=0)
 if not letters:return {'*':tuple(p.rect)}
 rows=[]
 for l in sorted(letters,key=lambda q:(q['bbox'][1],q['bbox'][0])):
  if not rows or abs(l['bbox'][1]-rows[-1][0]['bbox'][1])>15:rows.append([l])
  else:rows[-1].append(l)
 for row in rows:row.sort(key=lambda q:q['bbox'][0])
 native=bool(p.metadata if False else doc.metadata.get('keywords',''))
 allsp=spans(p);draw=p.get_drawings();out={}
 for rid,row in enumerate(rows):
  yy=row[0]['bbox'][1]; y0=0 if rid==0 else yy-5
  y1=rows[rid+1][0]['bbox'][1]-5 if rid+1<len(rows) else p.rect.height
  # Legacy letters are above the axes instead of the complete decorated panel.
  firstx=row[0]['bbox'][0]
  edges=[0.]+[max(0,x['bbox'][0]-(3 if native else firstx)) for x in row[1:]]+[p.rect.width]
  for cid,l in enumerate(row):
   clip=fitz.Rect(edges[cid],y0,edges[cid+1],y1)
   # Trim only empty margins; maintain the declared panel partition.
   ink=[]
   for s in allsp:
    r=fitz.Rect(s['bbox']);cx=(r.x0+r.x1)/2;cy=(r.y0+r.y1)/2
    if clip.contains(fitz.Point(cx,cy)):
     overflow=max(clip.x0-r.x0,r.x1-clip.x1,clip.y0-r.y0,r.y1-clip.y1,0)
     ink.append(r if overflow<=12 else r&clip)
   for d in draw:
    if d.get('fill') is not None and all(abs(v-1)<1e-8 for v in d['fill']) and (d.get('color') is None or all(abs(v-1)<1e-8 for v in d['color'])):continue
    r=fitz.Rect(d['rect'])
    if r.width>p.rect.width*.97 and r.height>p.rect.height*.97:continue
    if r.intersects(clip):ink.append(r&clip)
   for image in p.get_image_info():
    r=fitz.Rect(image['bbox'])
    if r.intersects(clip):ink.append(r&clip)
   if ink:
    t=union(ink); t=fitz.Rect(t.x0-2,t.y0-2,t.x1+2,t.y1+2)&p.rect
   else:t=clip
   out[l['letter']]=tuple(t)
 return out


def partitions(n,max_count=3):
 if n==0:yield []
 for k in range(1,min(max_count,n)+1):
  for rest in partitions(n-k,max_count):yield [k]+rest

def layout(items,legend_h=0.,counts=None):
 """One uniform paste scale for the whole figure, at most PASTE_SCALE.

 The scale is 1.0 whenever the frozen panel inventory fits the canvas and the
 540 pt cap; otherwise it is the single largest scale that does, so the type
 ladder inside one figure stays uniform (SI_PLAN 5.1 / review 4d).
 """
 best=None
 for counts in ([counts] if counts is not None else partitions(len(items))):
  start=0;rows=[];swidth=PASTE_SCALE
  for count in counts:
   sub=items[start:start+count];start+=count
   rows.append(sub)
  widths=column_widths(rows)
  for sub in rows:
   count=len(sub);width=sum(widths[count,c] for c in range(count))
   rowgutter=GUTTER3 if count==3 else GUTTER
   swidth=min(swidth,(W-2*MARGIN-rowgutter*(count-1)-LETTER_GUTTER*count)/width)
  base=sum(max(q['height'] for q in sub) for sub in rows)+legend_h
  chrome=LETTER_BAND*len(rows)+GUTTER*(len(rows)-1)+2*MARGIN
  sheight=(HEIGHT_CAP-chrome)/base if base>0 else PASTE_SCALE
  scale=min(swidth,max(sheight,MIN_SCALE),PASTE_SCALE)
  if scale<MIN_SCALE:continue
  height=base*scale+chrome
  penalty=(PASTE_SCALE-scale)*1e6+max(0,height-HEIGHT_CAP)*20+height*.05+sum(4 for n in counts if n==3)
  if best is None or penalty<best[0]:
   best=(penalty,[(sub,scale,max(q['height'] for q in sub)*scale+LETTER_BAND) for sub in rows],height,scale)
 if best is None:raise ValueError('no legal layout')
 return best[1],best[2],best[3]

def old_captions():
 cache=HERE/'original_captions.json'
 if cache.exists():return json.loads(cache.read_text())
 raise FileNotFoundError('Frozen original_captions.json is required; current captions are not an equivalent input.')

def clean_caption(s):
 for old,new in [
 ('This is the unchanged original 20-seed confirmation experiment, retained after replacing its main-text figure. ',''),
 ('The original route-generated capacity values are retained in','Route-generated capacity values are given in'),
 ('All outcomes and baseline definitions remain included. ',''),
 ('unchanged archived nonlinear fits','matched nonlinear fits'),
 ('These are the unchanged original','These are the original'),
 ('original 16-conductance model','16-conductance model'),
 ('Rates in \\textbf{A--D} are 0.003, 0.01, 0.03 and 0.01, respectively, frozen using five development seeds. Seeds vary permutations, initialization and noisy observations of fixed templates.','Rates were fixed on five development seeds; seeds vary permutations, initialization and noisy observations.'),
 ('The two axes have different ranges. ',''),
 ('All rates, 6,720 fresh fits and 53,760 checkpoints are retained; statistical scopes were fixed before training.','All 6,720 fits and 53,760 checkpoints are retained.'),
 ('All 2,240 ALS fits are retained. ',''),
 ('All 480 fits are retained. ',''),
 ('Both cohorts are independent of Supplementary Fig.~\\ref{fig:supp_morphology_credit}.','Both cohorts are independent of Supplementary Fig.~\\ref{fig:si_oracle_profile_credit}.'),
 # 2026-09-10: provenance coordinates carried inside frozen whole-sheet
 # captions, rewritten to the frozen supplement numbering (SI_NUMBERING 3).
 ('Supplementary Fig.~S45D,E','Supplementary Fig.~S8B,C'),
 ('the normalized dose response is in Supplementary Fig.~S48','the normalized dose response is in main Fig.~8D'),
 ("are main Fig.~4H's contrast",'give the leaf-assignment contrast reported in the main text'),
 ('The same condition as main Fig.~4E, repeated as the reference','The nested-target condition reported in the main text, shown as the reference'),
 ('neuron-specific','per-neuron'),
 ('matched-width (MW) scalar-fallback','matched-width scalar fallback'),
 ]:s=s.replace(old,new)
 return s

PANEL_MARK=re.compile(r'\\textbf\{([A-H](?:--[A-H])?(?:,[A-H])*)\}')
def panel_sentences(caption):
 """Split a frozen caption into one sentence group per panel letter."""
 out={};marks=list(PANEL_MARK.finditer(caption))
 for i,m in enumerate(marks):
  stop=marks[i+1].start() if i+1<len(marks) else len(caption)
  text=caption[m.end():stop].lstrip(', ').strip()
  group=m.group(1)
  letters=[]
  if '--' in group:
   a,b=group.split('--');letters=[chr(c) for c in range(ord(a),ord(b)+1)]
  else:letters=[q for q in group.split(',') if q]
  for letter in letters:out.setdefault(letter,text)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only');ap.add_argument('--no-audit',action='store_true');args=ap.parse_args()
 OUT.mkdir(exist_ok=True,parents=True);CFG.mkdir(exist_ok=True,parents=True);TEX.mkdir(exist_ok=True,parents=True)
 originals=old_captions();oldprov=json.loads((HERE/'original_provenance.json').read_text())
 bounds={k:make_bounds(k) for k in REG}
 for (key,panel),rect in PANEL_BOUNDS.items():bounds[key][panel]=rect
 (CFG/'source_panel_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
 bold=font_manager.findfont(font_manager.FontProperties(family=JS.SANS_FAMILY,weight='bold'))
 book=font_manager.findfont(font_manager.FontProperties(family=JS.SANS_FAMILY))
 assets=[];mapping={k:[] for k in REG};captions={};scales={}
 for number,(ident,module,groups,title,caption) in zip(CONSOLIDATED_FIGURE_NUMBERS, FIGURES, strict=True):
  dest=OUT/(ident+'.pdf'); label='fig:si_'+ident
  sources=[k for k,_ in groups]
  for key in sources:
   assert sha(J/REG[key]['path'])==REG[key]['sha256'],f'Changed source: {key}'
  whole=len(groups)==1 and groups[0][1]=='*'
  items=[]
  for key,letters in groups:
   if letters=='*':letters='*' if not REG[key]['letters'] else ''.join(sorted(q['letter'] for q in REG[key]['letters']))
   for oldletter in letters:
    bbox=fitz.Rect(bounds[key][oldletter])
    if not whole:bbox=trimmed_panel_bounds(key,oldletter,bbox)
    items.append({'source':key,'source_panel':oldletter,'bbox':list(bbox),'width':bbox.width,'height':bbox.height})
  panelmap=[];issues=[];ancillary=[];scale=1.0;letter_layout=None
  if ident in WHOLE_CROPS:
   key=groups[0][0];source=fitz.open(J/REG[key]['path']);clip=fitz.Rect(WHOLE_CROPS[ident]);output=fitz.open();p=output.new_page(width=clip.width,height=clip.height);p.show_pdf_page(p.rect,source,0,clip=clip)
   for item in items:panelmap.append(dict(item,panel=item['source_panel'],target_rect=item['bbox'],scale=1.0))
  elif whole:
   key=groups[0][0];output=fitz.open(J/REG[key]['path'])
   for item in items:
    panelmap.append(dict(item,panel=item['source_panel'],target_rect=item['bbox'],scale=1.0))
  else:
   legend_specs=SHARED_LEGENDS.get(ident,[])
   legend_raw=sum((r[3]-r[1]) for _,r in legend_specs)
   rows,height,scale=layout(items,legend_h=legend_raw,counts=ROW_COUNTS[ident])
   legend_h=legend_raw*scale+10*len(legend_specs)+(22 if ident in NATIVE_LEGENDS else 0)
   height=height-legend_raw*scale+legend_h
   output=fitz.open();p=output.new_page(width=W,height=height)
   p.insert_font(fontname='PanelSans',fontfile=bold);p.insert_font(fontname='ReflowAxis',fontfile=book)
   letter_layout={'schema':'panel-letter-layout/1','panels':[]}
   widths=column_widths([sub for sub,_,_ in rows])
   max_width=max(sum(widths[len(sub),c]*scale+LETTER_GUTTER for c in range(len(sub)))
                 +(GUTTER3 if len(sub)==3 else GUTTER)*(len(sub)-1)
                 for sub,_,_ in rows)
   y=MARGIN;nextletter=0
   for row_index,(sub,rowscale,rowh) in enumerate(rows):
    rowgutter=GUTTER3 if len(sub)==3 else GUTTER
    x=(W-max_width)/2
    for col_index,q in enumerate(sub):
     key=q['source'];oldletter=q['source_panel'];source=prepared_panel(key,oldletter);sp=source[0]
     # Remove only the source's original panel letters; underlying vectors remain.
     # fill=None: a filled redaction box is painted as a STROKED rectangle at the
     # PDF default width of 1.0 pt, which is not a line-weight token and put one
     # stray path per redaction into every cropped sheet.  The text still goes;
     # the regions that must also hide line art are masked below, unstroked.
     for patch in PANEL_PATCHES.get((key,oldletter),[]):
      ancillary.append(dict(patch,target_source_panel=key+oldletter,source_asset=REG[patch['source']]['path'],source_sha256=REG[patch['source']]['sha256']))
     clip=fitz.Rect(q['bbox']);target=fitz.Rect(x+LETTER_GUTTER,y+LETTER_BAND,x+LETTER_GUTTER+clip.width*scale,y+LETTER_BAND+clip.height*scale)
     p.show_pdf_page(target,source,0,clip=clip)
     for tt in PANEL_TEXT.get((key,oldletter),[]):
      origin=[target.x0+(tt['origin'][0]-clip.x0)*scale,target.y0+(tt['origin'][1]-clip.y0)*scale];p.insert_text(origin,tt['text'],fontname='ReflowAxis',fontsize=tt['fontsize']*scale,rotate=tt.get('rotate',0));ancillary.append(dict(tt,target_source_panel=key+oldletter,target_origin=origin))
     newletter=chr(65+nextletter);nextletter+=1
     p.insert_text((x,y+10.4),newletter,fontname='PanelSans',fontsize=JS.PT_LETTER,color=(.1,.12,.13))
     letter_layout['panels'].append({'letter':newletter,'row':row_index,
       'column':'leading' if col_index==0 else f'{len(sub)}_col{col_index}',
       'content_bbox':list(target)})
     panelmap.append(dict(q,panel=newletter,target_rect=list(target),scale=scale))
     x+=LETTER_GUTTER+widths[len(sub),col_index]*scale+rowgutter
    y+=rowh+GUTTER
   for key,r in legend_specs:
    source=fitz.open(J/REG[key]['path']);sp=source[0]
    # fill=None for the same reason as the panel crops above: a filled redaction
    # box is painted as a stroked rectangle at the 1.0 pt PDF default.
    for letter in REG[key]['letters']:sp.add_redact_annot(letter['bbox'],fill=None)
    if REG[key]['letters']:sp.apply_redactions(images=0,graphics=0,text=0)
    clip=fitz.Rect(r);cw=clip.width*scale;ch=clip.height*scale
    target=fitz.Rect((W-cw)/2,y-7,(W+cw)/2,y-7+ch);p.show_pdf_page(target,source,0,clip=clip)
    ancillary.append({'role':'shared legend','source':key,'source_asset':REG[key]['path'],'source_sha256':REG[key]['sha256'],'bbox':list(clip),'target_rect':list(target),'scale':scale});y+=ch+10
   if ident in NATIVE_LEGENDS:
    x=105
    for entry in NATIVE_LEGENDS[ident]:
     yy=y+1;p.draw_line((x,yy-3),(x+11,yy-3),color=entry['color'],width=JS.LW_DATA);p.insert_text((x+15,yy),entry['label'],fontname='ReflowAxis',fontsize=JS.PT_BASE);ancillary.append(dict(entry,role='native shared legend',source='S14',origin=[x,yy]));x+=110
  scales[ident]=round(scale,4)
  if args.only is None or args.only==ident:
   source_metadata=json.loads(output.metadata.get('keywords') or '{}')
   output.set_metadata({'title':title or re.sub(r'\\textbf\{([^}]+)\}.*',r'\1',originals[groups[0][0]],flags=re.S),'author':'Safaai, Richards and Sabatini','creator':'supplement_consolidation/build.py; native vector-panel reflow','keywords':json.dumps({'schema':'native-vector-reflow/1','id':ident,'figure':f'S{number}','paste_scale':round(scale,4),'panels':panelmap,'source_layout':source_metadata if whole else None,'letter_layout':letter_layout},separators=(',',':'))})
   output.save(dest,garbage=4,deflate=True,no_new_id=True)
   output[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/(ident+'.png'))
  for p in panelmap:
   source=p['source'];p['source_asset']=REG[source]['path'];p['source_sha256']=REG[source]['sha256'];p['old_provenance_figure']='fig9' if source=='M9' else 'fig'+source
   records=[r for r in oldprov if r.get('figure')==p['old_provenance_figure']]
   p['old_provenance_entry_ids']=sorted(set(r.get('entry_id','') for r in records))
   p['numerical_source_paths']=sorted(set(r.get('source_path','') for r in records if 'source_data/' in r.get('source_path','') and not r.get('source_path','').endswith('.pdf')))
   explicit=EXPLICIT_NUMERICAL_SOURCES.get((source,p['source_panel']),[])
   p['numerical_source_paths']=sorted(set(p['numerical_source_paths'])|{'drafts/dendritic-local-learning/journal/'+q for q in explicit})
   if (source,p['source_panel']) in NUMERICAL_SOURCE_OVERRIDES:
    # Complete current panel mappings supersede figure-wide historical links.
    # The historical entry identifiers above remain available for lineage.
    explicit=NUMERICAL_SOURCE_OVERRIDES[(source,p['source_panel'])]
    p['numerical_source_paths']=['drafts/dendritic-local-learning/journal/'+q for q in sorted(explicit)]
   p['explicit_numerical_source_hashes']={q:sha(J/q) for q in explicit}
   if source in EXTRA_ASSETS:p['record_type']='native supplement panel';p['builder']=REG[source]['builder']
   if (source,p['source_panel'])==('S12','B'):p['record_type']='schematic'
   mapping[source].append({'figure':f'S{number}','label':label,'panel':p['panel'],'source_panel':p['source_panel'],'path':str(dest.relative_to(J))})
  cap=clean_caption((r'\textbf{'+title+'} '+caption) if title else originals[groups[0][0]])
  if ident in CAPTION_APPEND:cap=cap.rstrip()+' '+CAPTION_APPEND[ident]
  captions[ident]=cap
  assets.append({'figure':f'S{number}','id':ident,'module':module,'path':str(dest.relative_to(J)),'label':label,'panels':panelmap,'ancillary_elements':ancillary,'whole_source_sheet':whole,'paste_scale':round(scale,4),'width_pt':output[0].rect.width,'height_pt':output[0].rect.height,'sha256':sha(dest) if dest.exists() else None,'source_data_directories':['source_data/'+d for d in SOURCE_DATA_DIRS.get(ident,[])]})
  print(f'S{number:<3d} {ident:<30s} height {output[0].rect.height:7.1f}  scale {scale:.3f}  panels {len(panelmap)}')
 for key in mapping:
  if not mapping[key]:
   fallback={'S8':('note:credit_images','Historical fixed-budget noise outcomes are archived because the executed generator is unresolved; they are excluded from current task and depth claims.'),
             'S17':('tab:animal_credit','Animal-level estimates and inference are retained in the external signed-credit table and Section S11; the redundant descriptive figure remains archived.'),
             'S9':('fig:si_utility_signal_noise','Old S9A and S9C are re-rendered on one dimensionless axis as panel E of the merged utility figure; the remaining panels are recorded in omitted_panel_status.'),
             'S6':('note:conflict_ancestry','Static quadratic interference remains in the derivation; the original numerical illustration is archived.'),
             'S16':('note:interior_optimum','Designed and retrospective optimum comparisons remain numerically in the operator-theory note; the full diagnostic is archived.'),
             'S26':('tab:physical_reproducibility','Same-seed rerun outcomes and outliers remain in the physical-depth reproducibility table.'),
             'S48':('fig:focal','The normalized-dose comparison is promoted to main Fig. 8D and is not duplicated in the supplement.'),
             # 2026-09-12: frozen sheets replaced by native renders (N14, N15, N18) pasted whole; kept in the registry for provenance.
             'S43':('fig:si_boolean_capacity','Replaced by the native render N14, drawn from the same source tables and pasted whole as Supplementary Fig. S14.'),
             'S44':('fig:si_boolean_learning','Replaced by the native render N15, drawn from the same source tables and pasted whole as Supplementary Fig. S15.'),
             'S42':('fig:si_conductance_grouping','Replaced by the native render N18, drawn from the same source tables and pasted whole as Supplementary Fig. S18.'),
             # 2026-09-12: eleven more frozen sheets replaced by native renders pasted whole; kept in the registry for provenance.
             'S49':('fig:si_mnist_dictionary_geometry','Replaced by the native render N7, drawn from the same source tables and pasted whole as Supplementary Fig. S7.'),
             'S45':('fig:si_error_field_geometry','Replaced by the native render N8, drawn from the same source tables and pasted whole as Supplementary Fig. S8.'),
             'S36':('fig:si_scalar_tree_capacity','Replaced by the native render N12, drawn from the same source tables and pasted whole as Supplementary Fig. S12.'),
             'S37':('fig:si_scalar_tree_capacity','Replaced by the native render N12, drawn from the same source tables and pasted whole as Supplementary Fig. S12.'),
             'S40':('fig:si_oracle_profile_credit','Replaced by the native render N13, drawn from the same source tables and pasted whole as Supplementary Fig. S13.'),
             'S54':('fig:si_fixed_profile_budget','Replaced by the native render N16, drawn from the same source tables and pasted whole as Supplementary Fig. S16.'),
             'S55':('fig:si_credit_optimizer_controls','Replaced by the native render N17, drawn from the same source tables and pasted whole as Supplementary Fig. S17.'),
             'S50':('fig:si_conductance_precision','Replaced by the native render N19, drawn from the same source tables and pasted whole as Supplementary Fig. S19.'),
             'S51':('fig:si_conductance_optimization','Replaced by the native render N20, drawn from the same source tables and pasted whole as Supplementary Fig. S20.'),
             'S52':('fig:si_conductance_optimization','Replaced by the native render N20, drawn from the same source tables and pasted whole as Supplementary Fig. S20.'),
             'S53':('fig:si_local_gate_controls','Replaced by the native render N21, drawn from the same source tables and pasted whole as Supplementary Fig. S21.'),
             'S38':('fig:si_finite_horizon','Replaced by the native render N35, drawn from the same source tables and pasted whole as Supplementary Fig. S36.'),
             'S39':('fig:si_finite_horizon','Replaced by the native render N35, drawn from the same source tables and pasted whole as Supplementary Fig. S36.'),
             'S41':('fig:si_morphology_estimation','Replaced by the native render N36, drawn from the same source tables and pasted whole as Supplementary Fig. S37.'),
             # 2026-09-13: the last four sheets that pasted crops of frozen or training-script renders are pasted whole from native renders; the retired keys keep their provenance entries.
             'S46':('fig:si_utility_signal_noise','Replaced by the native render N3, drawn from the same source tables and pasted whole as Supplementary Fig. S3.'),
             'S5':('fig:si_utility_signal_noise','Panel E is redrawn in the native render N3 from the same alignment-controlled tables and pasted whole as Supplementary Fig. S3F; the training builder is never run.'),
             'X1':('fig:si_utility_signal_noise','Redrawn inside the native render N3 as Supplementary Fig. S3E; the separate component is no longer pasted.'),
             'X2':('fig:si_utility_signal_noise','Redrawn inside the native render N3 as Supplementary Fig. S3G; the separate component is no longer pasted.'),
             'S31':('fig:si_physical_architecture','Replaced by the native render N22, drawn from the same source tables and pasted whole as Supplementary Fig. S24.'),
             'S47':('fig:si_shunt_replication','Replaced by the native render N30, drawn from the same source tables and pasted whole as Supplementary Fig. S32D,E.'),
             'S22':('fig:si_measured_transfer_geometry','Replaced by the native render N31, drawn from the same source tables and pasted whole as Supplementary Fig. S33A.'),
             'M9':('fig:si_measured_transfer_geometry','Replaced by the native render N31, drawn from the same source tables and pasted whole as Supplementary Fig. S33B--D.'),
             'S56':('fig:si_measured_transfer_geometry','Replaced by the native render N31, drawn from the same calibration table and pasted whole as Supplementary Fig. S33E.'),
             'S11':('fig:si_shunt_sensitivity','Replaced by the native render N29, which ports the legacy panels to the canvas from the same source tables and is pasted whole as Supplementary Fig. S31.'),
             'S21':('fig:si_shunt_sensitivity','Replaced by the native render N29, drawn from the same source tables and pasted whole as Supplementary Fig. S31.')}.get(key)
   if not fallback:raise ValueError('No destination: '+key)
   mapping[key]=[{'kind':'table' if key in ['S17','S26'] else 'section','label':fallback[0],'note':fallback[1]}]
 manifest={'schema':'supplement-consolidation/1','selection_is_editorial':True,'numerical_results_changed':False,'builder':'scripts/supplement_consolidation/build.py','specification':'scripts/supplement_consolidation/specification.py','source_registry':'scripts/supplement_consolidation/original_assets.json','frozen_input_hashes':{str(p.relative_to(J)):sha(p) for p in [HERE/'original_assets.json',HERE/'original_captions.json',HERE/'original_provenance.json']},'paste_scale_target':PASTE_SCALE,'height_cap_pt':HEIGHT_CAP,'assets':assets,'old_to_new':mapping}
 (CFG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (CFG/'captions.json').write_text(json.dumps(captions,indent=2)+'\n')
 write_reference_map(mapping,originals,assets)
 # Semantic aliases preserve cross-document references to scientific comparisons.
 aliases={}
 for key,dests in mapping.items():
  if key in EXTRA_ASSETS:continue
  if key=='M9':continue
  if dests[0].get('figure'):aliases.setdefault(dests[0]['label'],[]).append(REG[key]['old_label'])
 for ident,extra in ALIAS_EXTRA.items():aliases.setdefault('fig:si_'+ident,[]).extend(extra)
 for k in aliases:aliases[k]=[a for a in dict.fromkeys(aliases[k]) if a not in ALIAS_BLACKLIST and a!=k]
 for module in sorted(set(f[1] for f in FIGURES)):
  text=[]
  for a in assets:
   if a['module']!=module:continue
   labels='\n'.join(r'\label{'+x+'}' for x in [a['label']]+aliases.get(a['label'],[]))
   text.append(r'''\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{'''+a['path'].removeprefix('figures/')+r'''}
\caption{'''+captions[a['id']]+r'''}
'''+labels+r'''
\end{figure}
''')
  (TEX/(module+'_figures.tex')).write_text('\n'.join(text))
 below={k:v for k,v in scales.items() if v<PASTE_SCALE-1e-9}
 # An exemption is a work queue entry, not a waiver: every figure that cannot
 # be pasted at 1.0 must name the builder that would remove the exemption.
 missing=sorted(set(below)-set(SCALE_EXEMPTIONS))
 if missing:raise AssertionError('below PASTE_SCALE with no SCALE_EXEMPTIONS entry: '+', '.join(missing))
 stale=sorted(set(SCALE_EXEMPTIONS)-set(below))
 report={'paste_scale_target':PASTE_SCALE,'height_cap_pt':HEIGHT_CAP,'scales':scales,
         'below_target_scale':below,
         'scale_exemptions':{k:dict(SCALE_EXEMPTIONS[k],paste_scale=below[k]) for k in below},
         'cleared_scale_exemptions':stale,
         'over_height_cap':{a['id']:a['height_pt'] for a in assets if a['height_pt']>HEIGHT_CAP+.5}}
 if not args.no_audit:
  from figure_canvas import audit_native_pdf
  audits={}
  for a in assets:
   violations=audit_native_pdf(J/a['path'],strict=True)
   kinds={}
   for v in violations:kinds[v.kind]=kinds.get(v.kind,0)+1
   audits[a['id']]={'figure':a['figure'],'violations':len(violations),'by_kind':kinds,
                    'text_floor_examples':[v.detail for v in violations if v.kind=='text-floor'][:2]}
  report['strict_audit']=audits
 (CFG/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n')
 if args.only is None:
  # Produce a compact visual contact sheet for independent editorial review.
  review=fitz.open()
  for start in range(0,len(assets),6):
   page=review.new_page(width=840,height=1060)
   for j,a in enumerate(assets[start:start+6]):
    x=10+(j%2)*415;y=10+(j//2)*348
    page.insert_text((x,y+12),a['figure']+' '+a['id'],fontsize=9)
    inp=fitz.open(J/a['path']);page.show_pdf_page(fitz.Rect(x,y+20,x+405,y+338),inp,0)
  review.save(CFG/'contact_sheet.pdf',garbage=4,deflate=True,no_new_id=True)


def write_reference_map(mapping,originals,assets):
 """Every source sheet, where each of its panels went, and why it did not.

 ``omitted_panel_status`` carries one record per dropped panel: what it
 showed, why it is not printed and where its numbers still are.  The
 assertion below is what stops the registry answering that question with
 boilerplate (SI_PLAN 5.5).
 """
 dest_by_figure={a['figure']:a for a in assets}
 out={}
 for key,rec in REG.items():
  dests=mapping[key]
  used={d.get('source_panel') for d in dests if d.get('figure')}
  letters=[q['letter'] for q in rec['letters']] or ['*']
  sentences=panel_sentences(originals.get(key,''))
  omitted={}
  for letter in letters:
   if letter in used:continue
   override=PANEL_CONTENT.get((key,letter))
   if override:omitted[letter]=dict(override)
   else:
    content=sentences.get(letter)
    if not content:
     raise AssertionError(f'no content record for dropped panel {key}{letter}; add it to specification.PANEL_CONTENT')
    reason=PANEL_REASONS.get(key,'Omitted from the curated supplement; the measurement is retained in Source Data and in the section that cites it.')
    numbers=sorted({r.get('source_path','') for r in json.loads((HERE/'original_provenance.json').read_text())
                    if r.get('figure')==('fig9' if key=='M9' else 'fig'+key) and 'source_data/' in r.get('source_path','')})
    omitted[letter]={'content':content,'reason':reason,'numbers_at':numbers[:4] or 'source archive'}
  out[key]={'old_label':rec['old_label'],
            'original_asset':rec['path'],'original_sha256':rec['sha256'],
            'primary_destination':dests[0].get('figure') or dests[0].get('label'),
            'panels':{d['source_panel']:{'figure':d['figure'],'panel':d['panel'],'path':d['path']}
                      for d in dests if d.get('figure')},
            'omitted_panel_status':omitted}
 (CFG/'reference_map.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
