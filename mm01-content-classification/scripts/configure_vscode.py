"""Configure VS Code for /usr/bin/python3 and workspace-local dependency installation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[2]

def main():
    packages=WORKSPACE/'.system-python-packages';directory=ROOT/'.vscode';directory.mkdir(exist_ok=True)
    terminal={'PYTHONPATH':str(packages),'PATH':str(packages/'bin')+':${env:PATH}','PIP_TARGET':str(packages)}
    path=directory/'settings.json';settings=json.loads(path.read_text()) if path.exists() else {}
    settings.update({'python.defaultInterpreterPath':'/usr/bin/python3','python.terminal.activateEnvironment':False,'python.testing.unittestEnabled':True,'python.testing.pytestEnabled':False,'python.testing.unittestArgs':['-v','-s','tests'],'python.envFile':'${workspaceFolder}/.env','python.analysis.extraPaths':['${workspaceFolder}/src',str(packages)],'terminal.integrated.env.osx':terminal,'files.exclude':{'**/__pycache__':True},'search.exclude':{'data/cache':True,'data/raw':True,'runs':True,'source_history':True}})
    path.write_text(json.dumps(settings,indent=2))
    base={'type':'debugpy','request':'launch','python':'/usr/bin/python3','cwd':'${workspaceFolder}','console':'integratedTerminal','justMyCode':True,'env':{'PYTHONPATH':str(packages),'HF_HOME':'${workspaceFolder}/data/cache/huggingface','TORCH_HOME':'${workspaceFolder}/data/cache/torch','HF_HUB_OFFLINE':'1','MPLCONFIGDIR':'${workspaceFolder}/data/cache/matplotlib'}}
    configs=[{**base,'name':'01 系统 Python 与 PyTorch 检查','program':'${workspaceFolder}/scripts/check_system_environment.py'},{**base,'name':'02 调试当前 Python 文件','program':'${file}'},{**base,'name':'03 第一阶段 CLIP 训练','program':'${workspaceFolder}/scripts/run_baselines.py','args':['--backend','clip','--output','runs/${input:runName}']},{**base,'name':'04 第一阶段 ResNet 训练','program':'${workspaceFolder}/scripts/run_baselines.py','args':['--backend','resnet','--output','runs/${input:runName}']},{**base,'name':'05 第二阶段少样本实验','program':'${workspaceFolder}/scripts/run_phase2.py','args':['--output','runs/${input:runName}']}]
    configs.append({**base,'name':'06 Apple GPU（MPS）检测','program':'${workspaceFolder}/scripts/check_mps.py'})
    configs.append({**base,'name':'07 MPS 图文训练（四个模型）','program':'${workspaceFolder}/scripts/train_torch.py','args':['--device','mps','--fresh-features','--output','runs/${input:runName}']})
    configs.append({**base,'name':'08 运行项目测试','program':'${workspaceFolder}/scripts/run_vscode_checks.py'})
    (directory/'launch.json').write_text(json.dumps({'version':'0.2.0','configurations':configs,'inputs':[{'id':'runName','type':'promptString','description':'新的实验目录名（每次使用不同名称）','default':'system-python-debug-01'}]},indent=2,ensure_ascii=False))
    (directory/'extensions.json').write_text(json.dumps({'recommendations':['ms-python.python','ms-python.debugpy','ms-python.vscode-pylance']},indent=2))
    (ROOT/'.env').write_text('PYTHONPATH='+str(packages)+'\nHF_HOME='+str(ROOT/'data/cache/huggingface')+'\nTORCH_HOME='+str(ROOT/'data/cache/torch')+'\nHF_HUB_OFFLINE=1\n')
    (ROOT.parent/'multimodal-project.code-workspace').write_text(json.dumps({'folders':[{'path':'mm01-content-classification'}],'settings':{'python.defaultInterpreterPath':'/usr/bin/python3','terminal.integrated.env.osx':terminal}},indent=2))
    print('Configured /usr/bin/python3 with dependencies in',packages)
if __name__=='__main__':main()
