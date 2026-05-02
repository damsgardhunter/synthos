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
/**
 * Returns the MPI wrapper command + args if QE_MPI_RANKS is set (>= 2).
 * Example with QE_MPI_RANKS=4: { cmd: "mpirun", args: ["-np", "4"] }.
 * Override launcher via QE_MPI_WRAPPER (e.g. "srun") and pass extra args
 * via QE_MPI_ARGS. The #1 throughput win for pw.x on a multi-core VM —
 * typically 4-8× faster for 5-15 atom SCF on 4-8 ranks.
 */
function getMpiWrapper(): { cmd: string; args: string[]; qeArgs: string[] } | null {
  const ranks = parseInt(process.env.QE_MPI_RANKS ?? "", 10);
  if (!Number.isFinite(ranks) || ranks < 2) return null;
  const cmd = process.env.QE_MPI_WRAPPER || "mpirun";
  const extra = (process.env.QE_MPI_ARGS || "").split(/\s+/).filter(Boolean);
  // QE-specific args go AFTER the binary (e.g. -nk 5 for k-point pooling)
  // QE_NPOOL sets -nk which splits k-points across MPI groups for ~2-4× speedup
  const npool = parseInt(process.env.QE_NPOOL ?? "", 10);
  const qeArgs: string[] = [];
  if (Number.isFinite(npool) && npool >= 2) {
    qeArgs.push("-nk", String(npool));
  }
  return { cmd, args: ["-np", String(ranks), ...extra], qeArgs };
}

export function spawnQE(
  binary: string,
  options: { cwd: string; stdio: ("pipe" | "inherit" | "ignore")[]; wslInputFile?: string },
): ChildProcess {
  const mpi = getMpiWrapper();
  if (IS_WINDOWS) {
    if (options.wslInputFile) {
      // Use bash -c with file redirection to bypass WSL stdin-pipe unreliability
      const mpiPrefix = mpi ? `${mpi.cmd} ${mpi.args.map(a => `'${a}'`).join(" ")} ` : "";
      const qeSuffix = mpi?.qeArgs.length ? ` ${mpi.qeArgs.join(" ")}` : "";
      const cmd = `${mpiPrefix}'${binary}'${qeSuffix} < '${options.wslInputFile}'`;
      return nodeSpawn("wsl.exe", ["-d", "Ubuntu", "--", "bash", "-c", cmd], {
        cwd: options.cwd,
        stdio: ["ignore", "pipe", "pipe"],
      });
    }
    const qeArgs = mpi?.qeArgs ?? [];
    const innerArgs = mpi ? [mpi.cmd, ...mpi.args, binary, ...qeArgs] : [binary];
    return nodeSpawn("wsl.exe", ["-d", "Ubuntu", "--", ...innerArgs], {
      cwd: options.cwd,
      stdio: options.stdio as any,
    });
  }
  if (mpi) {
    // mpirun args go before binary, QE args (-nk, -npool) go after
    // Result: mpirun -np 10 --allow-run-as-root pw.x -nk 5
    return nodeSpawn(mpi.cmd, [...mpi.args, binary, ...mpi.qeArgs], { cwd: options.cwd, stdio: options.stdio as any });
  }
  return nodeSpawn(binary, { cwd: options.cwd, stdio: options.stdio as any });
}
