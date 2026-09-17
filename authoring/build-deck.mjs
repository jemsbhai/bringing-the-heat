import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

// Uses the Codex artifact runtime. See authoring/README.md for setup.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const skill = process.env.SKILL_DIR;
const py = process.env.RUNTIME_PYTHON;
const runtimeModules = process.env.RUNTIME_NODE_MODULES;
for (const [key,value] of Object.entries({SKILL_DIR:skill,RUNTIME_PYTHON:py,RUNTIME_NODE_MODULES:runtimeModules})) {
  if (!value || !path.isAbsolute(value)) throw new Error(`Set ${key} to an absolute runtime path.`);
}
const runtimeRequire = createRequire(path.join(runtimeModules,'..','artifact-runtime-resolver.cjs'));
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const data = JSON.parse(await fs.readFile(path.join(root,'authoring/slides.json'),'utf8'));
if (data.length !== 22) throw new Error('The talk deck must retain its 22 slides.');
if (!data[0].deckUrl?.startsWith('https://')) throw new Error('Set the cover deckUrl to its verified public Google URL before building.');
const buildId = new Date().toISOString().replaceAll(':','').replaceAll('.','');
const buildDir = path.join(root,'.build',`deck-${buildId}`);
const renderDir = path.join(buildDir,'rendered');
await fs.mkdir(renderDir,{recursive:true});
await fs.mkdir(path.join(buildDir,'finalized'),{recursive:true});
const deck = Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#151820',fg:'#FFF9ED',muted:'#B9BECA',yellow:'#FFD21E',orange:'#FF9D55',table:'#20242E'};
const family='Arial', mono='Consolas';

function text(s,t,x,y,w,h,size=30,color=C.fg,bold=false,font=family){
  const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  q.text=t;
  q.text.style={fontSize:size,typeface:font,color,bold,autoFit:'none',wrap:true,verticalAlignment:'top'};
  return q;
}
async function image(s,file,x,y,w,h,alt){
  s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,file))),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function footer(s,d,n){
  if(d.bottom)text(s,d.bottom,64,625,1120,48,22,C.muted);
  text(s,String(n).padStart(2,'0'),1200,665,45,28,18,C.muted);
}
function columns(s,d,y=205){
  text(s,d.leftTitle,64,y,530,48,32,C.yellow,true);
  text(s,d.left,64,y+66,526,Math.min(342,604-y-66),30);
  text(s,d.rightTitle,694,y,522,48,32,C.yellow,true);
  text(s,d.right,694,y+66,522,Math.min(342,604-y-66),30);
}
function table(s,d){
  const values=[d.headers,...d.rows];
  const cols=d.headers.length;
  const t=s.tables.add({rows:values.length,columns:cols,left:64,top:178,width:1152,height:d.kind==='measured'?304:394,columnWidths:cols===4?[350,230,286,286]:[285,405,462],values});
  t.borders.assign({fill:C.bg,width:0,style:'solid'});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:cols}).assign({fill:C.bg,textStyle:{typeface:family,fontSize:26,color:C.fg},margins:{left:14,right:16,top:10,bottom:10},anchor:'center'});
  t.cells.block({row:0,column:0,rowCount:1,columnCount:cols}).assign({fill:C.table,textStyle:{typeface:family,fontSize:27,color:C.yellow,bold:true}});
  for(let r=1;r<values.length;r++)if(r%2===0)t.cells.block({row:r,column:0,rowCount:1,columnCount:cols}).fill=C.table;
}
for(let i=0;i<data.length;i++){
  const d=data[i],s=deck.slides.add();s.background.fill=C.bg;
  if(d.kind==='cover'){
    text(s,d.name,76,245,702,112,78,C.yellow,true);
    const website=text(s,d.websiteLabel,80,380,660,70,36,C.fg);
    website.text.get(d.websiteLabel).link={uri:d.websiteUrl,isExternal:true};
    await image(s,'assets/deck-qr.png',820,164,392,392,`QR code for this slide deck: ${d.deckUrl}`);
  }else if(d.kind==='qa'){
    text(s,d.title,64,110,1100,120,96,C.yellow,true);
    text(s,d.subtitle,68,292,1120,180,52);
    footer(s,d,i+1);
  }else{
    text(s,d.title,64,54,1152,106,46,C.fg,true);
    if(d.kind==='columns'||d.kind==='closing')columns(s,d,d.kind==='closing'?204:188);
    else if(d.kind==='personal'){
      text(s,d.leftTitle,64,185,680,48,32,C.yellow,true);
      text(s,d.left,64,248,654,318,30);
      await image(s,'showcase/assets/multispecqr-rgb-test-row0.png',812,176,376,376,'Unchanged pixels of an actual MultiSpecQR RGB dataset sample');
      text(s,'Real RGB dataset sample\n3 payload layers',812,563,378,60,23,C.muted);
    }else if(d.kind==='table')table(s,d);
    else if(d.kind==='measured'){
      table(s,d);
      text(s,d.conditions,64,514,1152,94,24,C.muted);
    }
    else if(d.kind==='code'){
      text(s,d.code,64,181,700,417,25,C.fg,false,mono);
      text(s,d.rightTitle,826,186,385,80,31,C.yellow,true);
      text(s,d.right,826,277,385,322,28);
    }else if(d.kind==='lora'){
      text(s,d.formula,64,190,1152,135,86,C.yellow,true,mono);
      columns(s,d,379);
    }else if(d.kind==='steps'){
      d.items.forEach((it,j)=>{
        const y=185+j*133;
        text(s,it[0],64,y,104,77,58,C.yellow,true);
        text(s,it[1],197,y+4,1004,46,34,C.fg,true);
        text(s,it[2],197,y+60,1004,46,27,C.muted);
      });
    }else if(d.kind==='extras'||d.kind==='resources'){
      d.items.forEach((it,j)=>{
        const y=180+j*81;
        text(s,it[0],64,y,428,63,28,C.yellow,true);
        text(s,it[1],527,y,687,63,d.kind==='resources'?27:28,C.fg);
      });
    }
    footer(s,d,i+1);
  }
  s.speakerNotes.textFrame.setText(`TIMING: ${d.time}\n\n${d.notes}\n\nSOURCES\n${d.sources.join('\n')}`);
}

const candidate=path.join(buildDir,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
console.log('Draft exported');
for(let i=0;i<deck.slides.items.length;i++){
  const s=deck.slides.items[i];
  const png=await deck.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(renderDir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
  const layout=await s.export({format:'layout'});
  await fs.writeFile(path.join(renderDir,`slide-${String(i+1).padStart(2,'0')}.json`),await layout.text());
}
console.log('All slide previews rendered');
const out=path.join(buildDir,'finalized','Bringing-the-Heat.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:out,pythonExecutable:py,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[4,11,13,14,20,21].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:22,requiredNativeTableOwnerSlides:[4,11,13,14,20,21],fontPolicy:{basis:'design',families:[family,mono]},verifyArtifactToolImport:true,receiptPath:path.join(buildDir,'validation.json')});
console.log(JSON.stringify(result));

// Inspect exports from the finalized PPTX as well as the authoring previews.
const finalRenderDir=path.join(buildDir,'final-rendered');
await fs.mkdir(finalRenderDir,{recursive:true});
const finalizedDeck=await PresentationFile.importPptx(await FileBlob.load(out));
for(let i=0;i<finalizedDeck.slides.items.length;i++){
  const png=await finalizedDeck.export({slide:finalizedDeck.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(finalRenderDir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('Final PowerPoint slide previews rendered');

let guide='# Presenter guide\n\n**Bringing the Heat: Supercharging Your ML Pipelines with Hugging Face**\n\n30 minutes plus 5 minutes for Q&A. Slides 1–17 are the talk, slide 18 is Q&A, and slides 19–22 are optional appendix. Audience: students and ML engineers.\n\n## Run of show\n\n| Slide | Time | Topic |\n|---|---|---|\n';
for(let i=0;i<data.length;i++)guide+=`| ${i+1} | ${data[i].time} | ${data[i].title} |\n`;
guide+='\n## Speaker notes\n\n';
for(let i=0;i<data.length;i++){const d=data[i];guide+=`### ${i+1}. ${d.title} (${d.time})\n\n${d.notes}\n\n${d.sources.map(u=>`- <${u}>`).join('\n')}\n\n`;}
await fs.writeFile(path.join(buildDir,'presenter-guide.md'),guide);
console.log('Presenter guide written');

// Portable, offline rendering fallback. Arrow keys advance; N toggles notes.
const esc=t=>t.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const imgs=[];
for(let i=0;i<data.length;i++){
  const bytes=await fs.readFile(path.join(renderDir,`slide-${String(i+1).padStart(2,'0')}.png`));
  const coverLinks=i===0?`<a class="cover-link website" aria-label="Muntaser Syed website" href="${esc(data[i].websiteUrl)}" target="_blank" rel="noopener"></a><a class="cover-link deck" aria-label="Open the Google slide deck" href="${esc(data[i].deckUrl)}" target="_blank" rel="noopener"></a>`:'';
  imgs.push(`<section ${i?'hidden':''}><div class="slide-frame"><img alt="${esc(data[i].title)}" src="data:image/png;base64,${bytes.toString('base64')}">${coverLinks}</div><aside hidden><b>${data[i].time}</b><p>${esc(data[i].notes).replaceAll('\n','<br>')}</p></aside></section>`);
}
await fs.writeFile(path.join(buildDir,'slides-offline.html'),`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Bringing the Heat</title><style>body{margin:0;background:#151820;color:#fff9ed;font:18px Arial}section{width:100vw;height:100vh}section[hidden]{display:none}.slide-frame{position:absolute;width:min(100vw,177.7778vh);height:min(100vh,56.25vw);left:50%;top:50%;transform:translate(-50%,-50%)}section img{display:block;width:100%;height:100%;object-fit:contain}.cover-link{position:absolute;display:block}.cover-link:focus-visible{outline:3px solid #ffd21e}.cover-link.website{left:6.25%;top:52.778%;width:51.5625%;height:9.722%}.cover-link.deck{left:64.0625%;top:22.778%;width:30.625%;height:54.444%}aside{position:fixed;bottom:30px;left:4vw;right:4vw;background:#151820f5;padding:22px;border:1px solid #ffd21e;max-height:36vh;overflow:auto}nav{position:fixed;bottom:4px;right:16px;background:#151820;font-size:13px;color:#b9beca}button{background:transparent;color:inherit;border:0;font:inherit;cursor:pointer}@media print{section{display:block!important;page-break-after:always}.slide-frame{position:relative;width:100%;height:auto;left:0;top:0;transform:none}section img{height:auto}.cover-link{display:none}aside,nav{display:none!important}@page{size:landscape;margin:0}}</style>${imgs.join('')}<nav><button id="prev">Previous</button> <span id="count">1 / 22</span> <button id="next">Next</button> <button id="notes">Notes (N)</button> <button id="full">Full screen</button></nav><script>let i=0;const slides=[...document.querySelectorAll('section')];function go(n){slides[i].hidden=true;i=Math.max(0,Math.min(slides.length-1,n));slides[i].hidden=false;document.querySelector('#count').textContent=(i+1)+' / '+slides.length}function notes(){const a=slides[i].querySelector('aside');a.hidden=!a.hidden}document.querySelector('#prev').onclick=()=>go(i-1);document.querySelector('#next').onclick=()=>go(i+1);document.querySelector('#notes').onclick=notes;document.querySelector('#full').onclick=()=>document.documentElement.requestFullscreen();document.onkeydown=e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(i+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(i-1)}if(e.key.toLowerCase()==='n')notes();if(e.key==='Home')go(0);if(e.key==='End')go(slides.length-1)};</script></html>`);
console.log('Offline slide fallback written');

await fs.writeFile(path.join(buildDir,'build-manifest.json'),JSON.stringify({deck:out,previews:renderDir,finalPreviews:finalRenderDir,presenterGuide:path.join(buildDir,'presenter-guide.md'),offlineSlides:path.join(buildDir,'slides-offline.html'),receipt:path.join(buildDir,'validation.json')},null,2)+'\n');
console.log(`Review the files in ${buildDir} before copying them to output/.`);
