"""Download official author-linked public metadata; never the large image archive."""
import hashlib, json, sys, urllib.parse, urllib.request
from html.parser import HTMLParser
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT
FILES={'train':'1XsOkD3yhxgWu8URMes0S9LVSwuFJ4pT6','val':'1Z99QrwpthioZQY2U6HElmnx8jazf7-Kv','test':'1p9EewIKVcFbipVRLZNGYc2JbSC7A0SWv'}
class DownloadForm(HTMLParser):
    def __init__(self):super().__init__();self.params={}
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='input' and attrs.get('type')=='hidden':self.params[attrs['name']]=attrs.get('value','')
def valid(path):
    return path.exists() and 'clean_title\t' in path.open().readline() and path.stat().st_size>10000
def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True);manifest={}
    for split,id in FILES.items():
        target=raw/f'official-{split}.tsv'
        if not valid(target):
            url='https://drive.google.com/uc?'+urllib.parse.urlencode({'export':'download','id':id})
            response=urllib.request.urlopen(url,timeout=60)
            if 'text/html' in response.headers.get('Content-Type',''):
                form=DownloadForm();form.feed(response.read().decode());response.close()
                if form.params.get('id')!=id:raise ValueError('Unexpected download confirmation')
                response=urllib.request.urlopen('https://drive.usercontent.google.com/download?'+urllib.parse.urlencode(form.params),timeout=60)
            temp=target.with_suffix('.download')
            with response,temp.open('wb') as f:
                while chunk:=response.read(1024*1024):f.write(chunk)
            if not valid(temp):raise ValueError('Download was not TSV metadata')
            temp.replace(target)
        manifest[split]={'file_id':id,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        print(split,'ready',flush=True)
    (ROOT/'reports/official-source-files.json').write_text(json.dumps({'official_repository':'https://github.com/entitize/Fakeddit','official_folder':'https://drive.google.com/drive/folders/1DuH0YaEox08ZwzZDpRMOaFpMCeRyxiEF','files':manifest},indent=2))
if __name__=='__main__':main()
