import * as os from "os";
import * as path from "path";
import { spawn as nodeSpawn } from "child_process";
import type { ChildProcess, SpawnOptions } from "child_process";

export const IS_WINDOWS = process.platform === "win32";
export const EXE_EXT = IS_WINDOWS ? ".exe" : "";

/**
 * Returns the system temp directory joined with a subdirectory name.
 * Uses os.tmpdir() for cross-platform compatibility (avoids hardcoded /tmp).
 */
export function getTempSubdir(name: string): string {
  return path.join(os.tmpdir(), name);
}

/**
 * Returns the platform-appropriate shell and args for spawning shell commands.
 * On Windows: cmd.exe /C
 * On Unix: /bin/sh -c
 */
export function getShell(): { shell: string; shellFlag: string } {
  if (IS_WINDOWS) {
    return { shell: "cmd.exe", shellFlag: "/C" };
  }
  return { shell: "/bin/sh", shellFlag: "-c" };
}

/**
 * Gracefully kills a child process cross-platform.
 * On Windows, SIGTERM/SIGKILL are not supported — use process.kill() without signal.
 * On Unix, sends SIGTERM first then SIGKILL after a delay.
 */
export function killProcessGracefully(proc: ChildProcess, delayMs = 2000): void {
  if (!proc || proc.exitCode !== null) return;
  try {
    if (IS_WINDOWS) {
      proc.kill();
    } else {
      proc.kill("SIGTERM");
      setTimeout(() => {
        if (proc.exitCode === null) {
          try { proc.kill("SIGKILL"); } catch {}
        }
      }, delayMs);
    }
  } catch {}
}

/**
 * Adds .exe extension to binary path on Windows.
 */
export function binaryPath(p: string): string {
  if (IS_WINDOWS && !p.endsWith(".exe")) {
    return p + ".exe";
  }
  return p;
}

/**
 * Converts a Windows absolute path to a WSL /mnt/... path.
 * e.g. C:\Users\foo\bar → /mnt/c/Users/foo/bar
 */
export function toWslPath(winPath: string): string {
  return winPath
    .replace(/\\/g, "/")
    .replace(/^([A-Za-z]):/, (_, drive) => `/mnt/${drive.toLowerCase()}`);
}

/**
 * Spawns a QE binary cross-platform.
 * On Linux: spawn(binary, options)
 * On Windows: routes through wsl.exe so Linux QE binaries run inside WSL2 Ubuntu.
 *             WSL automatically translates the Windows cwd to the /mnt/... equivalent.
 *
 * When wslInputFile is provided (Windows only), uses bash -c with stdin redirection
 * instead of Node.js stdin piping — this avoids a known WSL issue where wsl.exe
 * does not reliably forward piped stdin from a Node.js parent process.
 */
export function spawnQE(
  binary: string,
  options: { cwd: string; stdio: ("pipe" | "inherit" | "ignore")[]; wslInputFile?: string },
): ChildProcess {
  if (IS_WINDOWS) {
    if (options.wslInputFile) {
      // Use bash -c with file redirection to bypass WSL stdin-pipe unreliability
      const cmd = `'${binary}' < '${options.wslInputFile}'`;
      return nodeSpawn("wsl.exe", ["-d", "Ubuntu", "--", "bash", "-c", cmd], {
        cwd: options.cwd,
        stdio: ["ignore", "pipe", "pipe"],
      });
    }
    return nodeSpawn("wsl.exe", ["-d", "Ubuntu", "--", binary], {
      cwd: options.cwd,
      stdio: options.stdio as any,
    });
  }
  return nodeSpawn(binary, { cwd: options.cwd, stdio: options.stdio as any });
}
