"""Verify debugpy launch, breakpoint, stack frame, step and completion on default Python."""
import json,os,queue,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    report=ROOT/'reports/system-python-debugger.json';stderr=(ROOT/'reports/system-debugger-stderr.log').open('w')
    proc=subprocess.Popen([sys.executable,'-m','debugpy.adapter'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,env=os.environ.copy())
    messages=queue.Queue();pending=[];seq=0;history=[]
    def reader():
        try:
            while True:
                headers={}
                while True:
                    line=proc.stdout.readline()
                    if not line:return
                    if line==b'\r\n':break
                    k,v=line.decode().split(':',1);headers[k.lower()]=v.strip()
                payload=proc.stdout.read(int(headers['content-length']));messages.put(json.loads(payload))
        except Exception as e:messages.put({'reader_error':str(e)})
    threading.Thread(target=reader,daemon=True).start()
    def send(command,arguments):
        nonlocal seq
        seq+=1;payload=json.dumps({'seq':seq,'type':'request','command':command,'arguments':arguments}).encode()
        proc.stdin.write(f'Content-Length: {len(payload)}\r\n\r\n'.encode()+payload);proc.stdin.flush();return seq
    def wait_for(predicate,timeout=40):
        deadline=time.time()+timeout
        while True:
            for i,m in enumerate(pending):
                if predicate(m):return pending.pop(i)
            left=deadline-time.time()
            if left<=0:raise TimeoutError('DAP wait timed out; last messages: '+str(history[-4:]))
            m=messages.get(timeout=left);history.append(m)
            if m.get('reader_error'):raise RuntimeError(m)
            if m.get('type')=='response' and not m.get('success',True):raise RuntimeError(m)
            pending.append(m)
    def response(n):return wait_for(lambda m:m.get('type')=='response' and m.get('request_seq')==n)
    try:
        response(send('initialize',{'adapterID':'python','clientID':'local-verification','linesStartAt1':True,'columnsStartAt1':True,'pathFormat':'path','supportsRunInTerminalRequest':False}))
        program=ROOT/'scripts/check_system_environment.py'
        launch=send('launch',{'name':'System Python debugger check','type':'debugpy','request':'launch','python':[sys.executable],'program':str(program),'args':['--report',str(ROOT/'reports/debug-runtime-confirmation.json')],'cwd':str(ROOT),'console':'internalConsole','justMyCode':True,'env':{'PYTHONPATH':os.environ['PYTHONPATH'],'MPLCONFIGDIR':str(ROOT/'data/cache/matplotlib')}})
        wait_for(lambda m:m.get('event')=='initialized')
        line=next(i for i,s in enumerate(program.read_text().splitlines(),1) if 'before=model.weight' in s)
        bp=response(send('setBreakpoints',{'source':{'path':str(program)},'breakpoints':[{'line':line}]}))
        assert bp['body']['breakpoints'][0]['verified']
        response(send('configurationDone',{}));response(launch)
        stopped=wait_for(lambda m:m.get('event')=='stopped');thread=stopped['body']['threadId']
        stack=response(send('stackTrace',{'threadId':thread}));frame=stack['body']['stackFrames'][0]['id']
        executable=response(send('evaluate',{'expression':'sys.executable','frameId':frame,'context':'watch'}))['body']['result']
        assert 'CommandLineTools' in executable or '/usr/bin/python3' in executable
        response(send('next',{'threadId':thread}));wait_for(lambda m:m.get('event')=='stopped')
        response(send('continue',{'threadId':thread}));wait_for(lambda m:m.get('event')=='terminated')
        actual=json.loads((ROOT/'reports/debug-runtime-confirmation.json').read_text());assert actual['pytorch_forward_backward_optimizer']=='passed'
        result={'adapter':'debugpy','breakpoint_verified':True,'stopped_in_source':str(program),'breakpoint_line':line,'python_evaluated_in_debugger':executable,'step_over':'passed','continued_to_completion':'passed','pytorch_forward_backward_optimizer':'passed'}
        report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill()
        stderr.close()
if __name__=='__main__':main()
