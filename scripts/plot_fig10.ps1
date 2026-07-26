# Overlay the run curves into results\fig10_reproduction.png
$python = "D:\Arbeit\Anaconda\envs\lwh\python.exe"
Set-Location "$PSScriptRoot\..\Learn-with-Helper"
& $python "$PSScriptRoot\plot_results.py" `
    --runs logs\lwh_prior logs\a3c_noprior logs\ddpg `
    --labels "LwH" "A3C (no prior)" "DDPG" `
    --out ..\results\fig10_reproduction.png
