# docker build --no-cache -t agentic-rag-api:v1.1.1.18 -f fastapi/Dockerfile fastapi
# docker save agentic-rag-api:v1.1.1.18 -o agentic-rag-api_v1.1.1.19.tar
# export-api-container.ps1
# Builds and exports agentic-rag-api with auto-incremented version from .version

$ErrorActionPreference = 'Stop'

# --- Settings you might tweak ---
$imageName   = 'agentic-rag-api'
$dockerfile  = Join-Path $PSScriptRoot 'fastapi/Dockerfile'
$buildCtx    = Join-Path $PSScriptRoot 'fastapi'
$versionFile = Join-Path $PSScriptRoot '.version'

function Get-VersionParts {
    param([string]$v)
    if ([string]::IsNullOrWhiteSpace($v)) {
        throw "Version string is empty."
    }
    $parts = $v.Trim() -split '\.'
    if ($parts.Count -lt 1) { throw "Invalid version format: '$v'." }
    $nums = @()
    foreach ($p in $parts) {
        if ($p -notmatch '^\d+$') { throw "Invalid numeric segment '$p' in version '$v'." }
        $nums += [int]$p
    }
    return ,$nums
}

function Increment-Version {
    param([int[]]$parts)

    # Base-100 for all but the most-significant is unbounded; carry propagates left.
    $i = $parts.Count - 1
    $parts[$i]++

    while ($i -gt 0 -and $parts[$i] -ge 100) {
        $parts[$i] = 0
        $i--
        $parts[$i]++
    }

    # If the most-significant (index 0) overflows 100, we DO NOT cap it (unbounded).
    return $parts
}

# --- Preconditions ---
if (-not (Test-Path $versionFile)) {
    throw "Missing $versionFile. Create it with an initial version like: 1.1.1.0"
}
if (-not (Test-Path $dockerfile)) { throw "Dockerfile not found at $dockerfile" }
if (-not (Test-Path $buildCtx))   { throw "Build context not found at $buildCtx" }

# --- Read and increment version ---
$currentVersion = (Get-Content -Raw $versionFile).Trim()
$parts = Get-VersionParts -v $currentVersion
$newParts = Increment-Version -parts $parts
$newVersion = ($newParts -join '.')

Write-Host "Current version : $currentVersion"
Write-Host "New version     : $newVersion"

# --- Docker build + save with new tag ---
$tag = "v$newVersion"
$imgRef = "${imageName}:${tag}"
$outTar = Join-Path $PSScriptRoot ("{0}_{1}.tar" -f $imageName, $tag)

Write-Host "Building image  : $imgRef"
docker build --no-cache -t $imgRef -f $dockerfile $buildCtx

Write-Host "Saving image to : $outTar"
docker save $imgRef -o $outTar

# --- Update .version for next build ---
Set-Content -Path $versionFile -Value $newVersion -Encoding ASCII
Write-Host "Updated $versionFile to $newVersion"

# --- Done ---
Write-Host "Build/export complete."
