param([string]$Python = '', [int]$Port = 8765, [string]$Config = '')
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $configured = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config\models.json') -Raw | ConvertFrom-Json
    if ($configured.python_executable) { $Python = $configured.python_executable }
    elseif (Test-Path -LiteralPath 'D:\Anaconda\CondaData\env_dirs\WDCNN\python.exe') { $Python = 'D:\Anaconda\CondaData\env_dirs\WDCNN\python.exe' }
    else { $Python = 'python' }
}
$serverArgs = @('-B', (Join-Path $PSScriptRoot 'scripts\server.py'), '--port', "$Port")
if ($Config) { $serverArgs += @('--config', $Config) }
& $Python @serverArgs
