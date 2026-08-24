import json
from pathlib import Path

def test_scheduler_examples_have_required_tasks_and_no_user_path():
    root=Path(__file__).resolve().parents[1]/'same_day_1445/live_optimizer';cfg=json.loads((root/'runtime/scheduler.example.json').read_text(encoding='utf-8'));assert cfg['tasks']['preclose']['time']=='14:45:30';assert cfg['tasks']['close']['time']=='15:10:00';assert cfg['tasks']['optimizer']['time']=='15:20:00';install=(root/'ops/install_windows_tasks.ps1').read_text(encoding='utf-8');remove=(root/'ops/remove_windows_tasks.ps1').read_text(encoding='utf-8');assert 'Register-ScheduledTask' in install;assert 'run_pipeline' in install;assert 'Unregister-ScheduledTask' in remove;assert 'Rosecccy' not in install and '10619' not in install
def test_scheduler_runs_weekdays_not_every_calendar_day():
    root=Path(__file__).resolve().parents[1]/'same_day_1445/live_optimizer';install=(root/'ops/install_windows_tasks.ps1').read_text(encoding='utf-8');assert '-Weekly' in install;assert 'Monday' in install and 'Friday' in install;assert 'New-ScheduledTaskTrigger -Daily' not in install
