export function LoadingAnimation({ className = "", size = "md" }: { className?: string; size?: "sm" | "md" | "lg" }) {
  const sizeClasses = {
    sm: "max-w-[80px] max-h-[80px]",
    md: "max-w-[120px] max-h-[120px]",
    lg: "max-w-[160px] max-h-[160px]",
  };

  return (
    <div className={`relative flex items-center justify-center w-full overflow-hidden rounded-lg ${className}`}
      style={{
        background: "radial-gradient(ellipse at center, hsl(220 10% 8%) 0%, hsl(220 10% 5%) 50%, hsl(220 10% 3%) 100%)",
      }}
    >
      {/* Star field background */}
      <div className="absolute inset-0 overflow-hidden">
        {Array.from({ length: 30 }).map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-white"
            style={{
              width: `${Math.random() * 2 + 0.5}px`,
              height: `${Math.random() * 2 + 0.5}px`,
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              opacity: Math.random() * 0.5 + 0.1,
              animation: `goldPulse ${2 + Math.random() * 4}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 3}s`,
            }}
          />
        ))}
      </div>

      {/* Centered video */}
      <div className={`relative z-10 ${sizeClasses[size]}`}>
        <video
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-contain"
          src="/loading.mp4"
        />
      </div>
    </div>
  );
}

export function LoadingCard({ height = "h-48", size = "md" }: { height?: string; size?: "sm" | "md" | "lg" }) {
  return (
    <div className={`${height} rounded-xl border border-[hsl(var(--gold)/0.15)]`}>
      <LoadingAnimation className="h-full" size={size} />
    </div>
  );
}
