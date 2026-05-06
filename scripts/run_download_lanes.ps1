param(
    [int]$Minutes = 60,
    [int]$DataverseJobs = 24,
    [int]$PhysioNetJobs = 32,
    [int]$NinaProJobs = 6,
    [int]$HttpFreshWorkers = 2,
    [string]$DataRoot = $env:MYOAGENCY_DATA_ROOT
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..").Path
$RawRoot = if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    Join-Path $Root "data\raw"
} else {
    Join-Path $DataRoot "raw"
}
$LogDir = Join-Path $Root "results\downloads\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $RawRoot | Out-Null

function Test-NonEmptyFile {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    return (Test-Path $Path) -and ((Get-Item $Path).Length -gt 0)
}

function Resolve-Queue {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if (Test-NonEmptyFile $candidate) {
            return $candidate
        }
    }
    return $null
}

function Start-Lane {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$LaneArgs
    )
    $stdout = Join-Path $LogDir "$Name.stdout.log"
    $stderr = Join-Path $LogDir "$Name.stderr.log"
    Write-Host "Starting $Name"
    $process = Start-Process -FilePath $Exe -ArgumentList $LaneArgs -WorkingDirectory $Root -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    [pscustomobject]@{
        Name = $Name
        Process = $process
        Stdout = $stdout
        Stderr = $stderr
    }
}

$lanes = @()
$aria = (Get-Command aria2c.exe -ErrorAction Stop).Source
$python = (Get-Command python.exe -ErrorAction Stop).Source

$dataverseQueue = Resolve-Queue @(
    (Join-Path $Root "results\downloads\split\dataverse.remaining"),
    (Join-Path $Root "results\downloads\split\dataverse.aria2.txt")
)
if (Test-NonEmptyFile $dataverseQueue) {
    $lanes += Start-Lane -Name "dataverse" -Exe $aria -LaneArgs @(
        "-i", $dataverseQueue,
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--max-concurrent-downloads=$DataverseJobs",
        "--split=8",
        "--max-connection-per-server=8",
        "--min-split-size=8M",
        "--file-allocation=none",
        "--retry-wait=5",
        "--max-tries=50",
        "--lowest-speed-limit=0",
        "--timeout=90",
        "--connect-timeout=30",
        "--summary-interval=300",
        "--console-log-level=warn",
        "--log-level=warn",
        "--save-session=$dataverseQueue",
        "--save-session-interval=60"
    )
}

$physionetQueue = Resolve-Queue @(
    (Join-Path $Root "results\downloads\split\physionet_s3.remaining"),
    (Join-Path $Root "results\downloads\split\physionet_s3.aria2.txt"),
    (Join-Path $Root "results\downloads\split\physionet.remaining"),
    (Join-Path $Root "results\downloads\split\physionet.aria2.txt")
)
if (Test-NonEmptyFile $physionetQueue) {
    $isS3 = $physionetQueue -like "*physionet_s3.remaining"
    $physionetLaneName = if ($isS3) { "physionet-s3" } else { "physionet" }
    $physionetSplit = if ($isS3) { 8 } else { 2 }
    $physionetConnections = if ($isS3) { 16 } else { 2 }
    $lanes += Start-Lane -Name $physionetLaneName -Exe $aria -LaneArgs @(
        "-i", $physionetQueue,
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--max-concurrent-downloads=$PhysioNetJobs",
        "--split=$physionetSplit",
        "--max-connection-per-server=$physionetConnections",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--retry-wait=10",
        "--max-tries=80",
        "--lowest-speed-limit=0",
        "--timeout=120",
        "--connect-timeout=40",
        "--summary-interval=300",
        "--console-log-level=warn",
        "--log-level=warn",
        "--save-session=$physionetQueue",
        "--save-session-interval=60"
    )
}

$ninaproQueue = Resolve-Queue @(
    (Join-Path $Root "results\downloads\split\archives_remaining_split\ninapro.aria2.txt"),
    (Join-Path $Root "results\downloads\split\ninapro.remaining"),
    (Join-Path $Root "results\downloads\split\ninapro.aria2.txt")
)
if (Test-NonEmptyFile $ninaproQueue) {
    $lanes += Start-Lane -Name "ninapro" -Exe $aria -LaneArgs @(
        "-i", $ninaproQueue,
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--max-concurrent-downloads=$NinaProJobs",
        "--split=2",
        "--max-connection-per-server=2",
        "--min-split-size=4M",
        "--file-allocation=none",
        "--retry-wait=10",
        "--max-tries=50",
        "--lowest-speed-limit=0",
        "--timeout=120",
        "--connect-timeout=40",
        "--summary-interval=300",
        "--console-log-level=warn",
        "--log-level=warn",
        "--save-session=$ninaproQueue",
        "--save-session-interval=60"
    )
}

$lanes += Start-Lane -Name "zenodo-fresh" -Exe $python -LaneArgs @(
    "scripts\download_http_fresh.py",
    "--root", $RawRoot,
    "--workers", "$HttpFreshWorkers",
    "--manifest", "results\downloads\zenodo_fresh_manifest.json",
    "--zenodo", "14224328:cemhsey",
    "--zenodo", "14272463:cemhsey"
)

$lanes += Start-Lane -Name "figshare-fresh" -Exe $python -LaneArgs @(
    "scripts\download_http_fresh.py",
    "--root", $RawRoot,
    "--workers", "$HttpFreshWorkers",
    "--manifest", "results\downloads\figshare_fresh_manifest.json",
    "--figshare", "7210397:1:capgmyo_dba",
    "--figshare", "13625828:1:seeds_65"
)

if ($lanes.Count -eq 0) {
    Write-Host "No non-empty queues found."
    exit 0
}

$deadline = (Get-Date).AddMinutes($Minutes)
while ((Get-Date) -lt $deadline) {
    $alive = @($lanes | Where-Object { -not $_.Process.HasExited })
    if ($alive.Count -eq 0) {
        break
    }
    $status = $alive | ForEach-Object { "$($_.Name):pid=$($_.Process.Id)" }
    Write-Host ("Running " + ($status -join " "))
    Start-Sleep -Seconds 60
}

foreach ($lane in $lanes) {
    if (-not $lane.Process.HasExited) {
        Write-Host "Stopping $($lane.Name) after $Minutes minute window"
        Stop-Process -Id $lane.Process.Id -Force
    }
}

$summary = foreach ($lane in $lanes) {
    $process = $lane.Process
    $process.Refresh()
    $exitCode = if ($process.HasExited) { $process.ExitCode } else { $null }
    [pscustomobject]@{
        Lane = $lane.Name
        ExitCode = $exitCode
        Stdout = $lane.Stdout
        Stderr = $lane.Stderr
    }
}
$summary | Format-Table -AutoSize

