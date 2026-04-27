interface LevelOneLoadMoreButtonProps {
  label: string;
  onClick: () => void;
}

export default function LevelOneLoadMoreButton({
  label,
  onClick,
}: LevelOneLoadMoreButtonProps) {
  return (
    <div className="sticky bottom-0 z-10 flex justify-center bg-gradient-to-t from-slate-800 via-slate-800/95 to-transparent pb-3 pt-6">
      <button
        type="button"
        onClick={onClick}
        className="rounded-full border border-white/30 bg-white/20 px-8 py-2.5 text-sm font-bold text-white shadow-lg backdrop-blur-md transition-colors hover:bg-white/30 active:scale-95"
      >
        {label}
      </button>
    </div>
  );
}
