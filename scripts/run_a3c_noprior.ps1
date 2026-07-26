# Thesis Fig 10 companion: plain A3C without the helper prior.
# Expected to stay near zero success under sparse reward (that is the point).
$python = "D:\Arbeit\Anaconda\envs\lwh\python.exe"
Set-Location "$PSScriptRoot\..\Learn-with-Helper"
& $python main.py `
    --workers 8 `
    --training-steps 1400000 `
    --cache-interval 50000 `
    --test-episodes 50 `
    --demo-type uav `
    --variance 0.1225 `
    --prior-decay 1e-5 `
    --log-dir logs\a3c_noprior
