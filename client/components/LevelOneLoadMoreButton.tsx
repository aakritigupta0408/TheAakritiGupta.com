interface LevelOneLoadMoreButtonProps {
  label: string;
  onClick: () => void;
  /** "dark" matches the slate content frame (default); "light" matches
   *  pages that override the frame to the light ground. */
  variant?: "dark" | "light";
}

export default function LevelOneLoadMoreButton({
  label,
  onClick,
  variant = "dark",
}: LevelOneLoadMoreButtonProps) {
  const isLight = variant === "light";
  return (
    <div
      className={`sticky bottom-0 z-10 flex justify-center pb-3 pt-6 ${
        isLight
          ? "bg-gradient-to-t from-[#f5f5f7] via-[#f5f5f7]/95 to-transparent"
          : "bg-gradient-to-t from-slate-800 via-slate-800/95 to-transparent"
      }`}
    >
      <button
        type="button"
        onClick={onClick}
        className={
          isLight
            ? "rounded-full border border-slate-300 bg-white px-8 py-2.5 text-sm font-bold text-slate-700 shadow-md transition-colors hover:border-slate-400 hover:bg-slate-50 active:scale-95"
            : "rounded-full border border-white/30 bg-white/20 px-8 py-2.5 text-sm font-bold text-white shadow-lg backdrop-blur-md transition-colors hover:bg-white/30 active:scale-95"
        }
      >
        {label}
      </button>
    </div>
  );
}
