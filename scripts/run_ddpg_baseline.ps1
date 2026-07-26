# Thesis Fig 10, curve 2: classical DDPG baseline (single process, slower).
$python = "D:\Arbeit\Anaconda\envs\lwh\python.exe"
Set-Location "$PSScriptRoot\..\Learn-with-Helper"
& $python ddpg_baseline.py `
    --training-steps 1400000 `
    --cache-interval 50000 `
    --test-episodes 50 `
    --log-dir logs\ddpg
