"""Install project dependencies for the default macOS Python, without a virtualenv."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[2]

def main():
    if sys.version_info[:2]!=(3,9):raise RuntimeError('This compatibility lock targets the existing system Python 3.9. Use its interpreter explicitly.')
    target=WORKSPACE/'.system-python-packages'
    subprocess.run([sys.executable,'-m','pip','install','--target',str(target),'--upgrade','--only-binary=:all:','--no-cache-dir','-r',str(ROOT/'requirements-system-python.txt'),'--log',str(ROOT/'reports/system-python-install.log')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/configure_vscode.py')],check=True)
    env=os.environ.copy();env['PYTHONPATH']=str(target);env['MPLCONFIGDIR']=str(ROOT/'data/cache/matplotlib')
    subprocess.run([sys.executable,str(ROOT/'scripts/check_system_environment.py'),'--report',str(ROOT/'reports/system-python-environment.json')],env=env,check=True)
if __name__=='__main__':main()
