# Thesis Fig 10, curve 1: LwH (A3C + SenAvo-Pri helper prior).
# ~10-20 min on a modern multicore CPU.
$python = "D:\Arbeit\Anaconda\envs\lwh\python.exe"
Set-Location "$PSScriptRoot\..\Learn-with-Helper"
& $python main.py `
    --workers 8 `
    --training-steps 1400000 `
    --cache-interval 50000 `
    --test-episodes 50 `
    --use-prior `
    --demo-type uav `
    --variance 0.1225 `
    --prior-decay 1e-5 `
    --log-dir logs\lwh_prior
