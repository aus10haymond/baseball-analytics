# PowerShell Script to Schedule Daily Fantasy Baseball Workflow
# Configures Windows Task Scheduler to run daily_workflow.py at 9 AM daily

param(
    [string]$PythonPath = "python",
    [string]$ScriptPath = "$PSScriptRoot\daily_workflow.py",
    [string]$Time = "09:00",
    [switch]$Remove
)

$TaskName = "FantasyBaseballDaily"
$Description = "Daily fantasy baseball projections and recommendations"

# Check if we're removing the task
if ($Remove) {
    Write-Host "Removing scheduled task '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Task removed successfully."
    exit 0
}

# Validate Python path
$pythonExists = Get-Command $PythonPath -ErrorAction SilentlyContinue
if (-not $pythonExists) {
    Write-Host "ERROR: Python not found at '$PythonPath'" -ForegroundColor Red
    Write-Host "Please provide the correct Python path using -PythonPath parameter"
    exit 1
}

# Validate script path
if (-not (Test-Path $ScriptPath)) {
    Write-Host "ERROR: Script not found at '$ScriptPath'" -ForegroundColor Red
    exit 1
}

Write-Host "Setting up daily workflow automation..." -ForegroundColor Green
Write-Host "  Task name: $TaskName"
Write-Host "  Python: $PythonPath"
Write-Host "  Script: $ScriptPath"
Write-Host "  Time: $Time daily"

# Remove existing task if it exists
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Create the action (run Python script)
$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "`"$ScriptPath`"" `
    -WorkingDirectory (Split-Path $ScriptPath)

# Create the trigger (daily at specified time)
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time

# Create task settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $Description `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -User $env:USERNAME `
        -Force

    Write-Host "`nTask scheduled successfully!" -ForegroundColor Green
    Write-Host "`nTask details:" -ForegroundColor Cyan
    Get-ScheduledTask -TaskName $TaskName | Format-List

    Write-Host "`nTo test the task immediately, run:" -ForegroundColor Yellow
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "`nTo remove the task, run:" -ForegroundColor Yellow
    Write-Host "  .\schedule_daily.ps1 -Remove"

} catch {
    Write-Host "`nERROR: Failed to create scheduled task" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}
