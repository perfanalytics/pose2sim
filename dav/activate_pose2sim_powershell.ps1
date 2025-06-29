
    # Change to the specified directory
    cd "D:\softs\github_david\pose2sim\dav"
    
    # Activate Conda environment and run script
    conda activate Pose2Sim; python run_pose2sim.py
    
    # Pause to keep the window open
    Write-Host "Press any key to continue..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    