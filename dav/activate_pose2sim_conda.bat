
    @echo off
    setlocal EnableDelayedExpansion
    
    REM Change to the specified directory
    cd "D:\softs\github_david\pose2sim\dav"
    
    REM Launch the Python script
    call conda activate Pose2Sim && python run_pose2sim.py
    
    REM Pause to keep the window open
    pause
    
    endlocal
    