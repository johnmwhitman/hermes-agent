/**
 * Regression coverage for optional native/WASI dependency records in the
 * committed npm lock graph.
 *
 * npm can install a platform-native package from an ancestor even when a
 * nested wrapper requires a different exact version, and platform-filtered
 * installs can hide missing WASI records.  Resolve the committed graph the
 * same way Node walks ancestor node_modules directories so those omissions
 * fail before a release archive is built.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

type LockPackage = {
  version?: string
  resolved?: string
  integrity?: string
  dependencies?: Record<string, string>
  devDependencies?: Record<string, string>
  optionalDependencies?: Record<string, string>
}

type Lockfile = {
  packages: Record<string, LockPackage>
}

type RootPackage = {
  allowScripts?: Record<string, boolean>
}

type DesktopPackage = {
  devDependencies?: Record<string, string>
}

const REPO_ROOT = path.resolve(__dirname, '..')

const lock = JSON.parse(
  fs.readFileSync(path.join(REPO_ROOT, 'package-lock.json'), 'utf-8')
) as Lockfile

const rootPackage = JSON.parse(
  fs.readFileSync(path.join(REPO_ROOT, 'package.json'), 'utf-8')
) as RootPackage

const desktopPackage = JSON.parse(
  fs.readFileSync(path.join(REPO_ROOT, 'apps', 'desktop', 'package.json'), 'utf-8')
) as DesktopPackage

function agentVersion(): string {
  const pyproject = fs.readFileSync(path.join(REPO_ROOT, 'pyproject.toml'), 'utf-8')
  const project = pyproject.match(/\[project\]\s+([\s\S]*?)(?=\n\[|$)/)?.[1]
  const version = project?.match(/^version\s*=\s*"([^"]+)"/m)?.[1]

  assert.ok(version, 'pyproject.toml [project] has no version')

  return version
}

function resolveLockedDependency(from: string, name: string): LockPackage | undefined {
  let scope = from

  while (true) {
    const nested = lock.packages[`${scope}/node_modules/${name}`]

    if (nested) {
      return nested
    }

    const ancestor = scope.lastIndexOf('/node_modules/')

    if (ancestor === -1) {
      break
    }

    scope = scope.slice(0, ancestor)
  }

  return lock.packages[`node_modules/${name}`]
}

function satisfies(version: string, requirement: string): boolean {
  if (!requirement.startsWith('^')) {
    return version === requirement
  }

  const want = requirement.slice(1).split('.').map(Number)
  const got = version.split('.').map(Number)

  if (want.length !== 3 || got.length !== 3 || [...want, ...got].some(Number.isNaN)) {
    return false
  }

  const [wantMajor, wantMinor, wantPatch] = want
  const [gotMajor, gotMinor, gotPatch] = got

  const atLeastMinimum =
    gotMajor > wantMajor ||
    (gotMajor === wantMajor && gotMinor > wantMinor) ||
    (gotMajor === wantMajor && gotMinor === wantMinor && gotPatch >= wantPatch)

  const belowUpperBound =
    wantMajor > 0
      ? gotMajor === wantMajor
      : wantMinor > 0
        ? gotMajor === 0 && gotMinor === wantMinor
        : gotMajor === 0 && gotMinor === 0 && gotPatch === wantPatch

  return atLeastMinimum && belowUpperBound
}

function assertCompleteDependencies(packagePath: string): void {
  const packageRecord = lock.packages[packagePath]

  assert.ok(packageRecord, `missing lock record for ${packagePath}`)

  for (const [name, requirement] of Object.entries(packageRecord.dependencies ?? {})) {
    const dependency = resolveLockedDependency(packagePath, name)

    assert.ok(dependency?.version, `${packagePath} cannot resolve ${name}@${requirement}`)
    assert.ok(
      satisfies(dependency.version, requirement),
      `${packagePath} resolves ${name}@${dependency.version}, which does not satisfy ${requirement}`
    )
  }
}

test('nested Lightning CSS resolves every exact platform binary at its own version', () => {
  const wrapperPath = 'node_modules/vite/node_modules/lightningcss'
  const wrapper = lock.packages[wrapperPath]

  assert.ok(wrapper, `missing lock record for ${wrapperPath}`)

  for (const [name, version] of Object.entries(wrapper.optionalDependencies ?? {})) {
    const native = resolveLockedDependency(wrapperPath, name)

    assert.equal(native?.version, version, `${wrapperPath} must resolve ${name}@${version}`)
    assert.match(native.resolved ?? '', /^https:\/\/registry\.npmjs\.org\//)
    assert.match(native.integrity ?? '', /^sha512-/)
  }
})

test('optional WASI bindings resolve complete semver-compatible dependency subtrees', () => {
  assertCompleteDependencies('node_modules/@rolldown/binding-wasm32-wasi')
  assertCompleteDependencies('node_modules/@tailwindcss/oxide-wasm32-wasi')
  assertCompleteDependencies('node_modules/@napi-rs/wasm-runtime')
})

test('Desktop Electron security floor uses the internal extractor graph', () => {
  const electronVersion = '41.10.3'
  const electron = lock.packages['node_modules/electron']

  assert.equal(desktopPackage.devDependencies?.electron, electronVersion)
  assert.equal(lock.packages['apps/desktop']?.devDependencies?.electron, electronVersion)
  assert.equal(electron?.version, electronVersion)
  assert.equal(electron?.dependencies?.['@electron-internal/extract-zip'], '^1.0.1')
  assert.equal(electron?.dependencies?.['@electron/get'], '^5.0.0')
  assert.equal(electron?.dependencies?.['extract-zip'], undefined)
  assert.equal(lock.packages['node_modules/extract-zip'], undefined)
  assertCompleteDependencies('node_modules/electron')
  assertCompleteDependencies('node_modules/@electron-internal/extract-zip')
  assert.equal(rootPackage.allowScripts?.[`electron@${electronVersion}`], true)
  assert.equal(rootPackage.allowScripts?.['electron@40.10.2'], undefined)
})

test('Desktop package and workspace lock metadata match the Agent release version', () => {
  const desktopPackage = JSON.parse(
    fs.readFileSync(path.join(REPO_ROOT, 'apps', 'desktop', 'package.json'), 'utf-8')
  ) as { version?: string }

  const releaseVersion = agentVersion()

  assert.equal(desktopPackage.version, releaseVersion)
  assert.equal(lock.packages['apps/desktop']?.version, releaseVersion)
})
