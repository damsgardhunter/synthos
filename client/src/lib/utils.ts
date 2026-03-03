import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function safeNum(val: unknown, fallback: number = 0): number {
  if (val === null || val === undefined) return fallback;
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

export function safeDisplay(val: unknown, digits?: number, fallback: string = "--"): string {
  if (val === null || val === undefined) return fallback;
  const n = Number(val);
  if (!Number.isFinite(n)) return fallback;
  return digits !== undefined ? n.toFixed(digits) : n.toLocaleString();
}
