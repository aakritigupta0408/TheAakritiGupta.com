import { motion } from "framer-motion";

interface LevelOneLoadMoreButtonProps {
  label: string;
  glowClassName: string;
  onClick: () => void;
}

export default function LevelOneLoadMoreButton({
  label,
  glowClassName,
  onClick,
}: LevelOneLoadMoreButtonProps) {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.96 }}
      transition={{ type: "spring", stiffness: 420, damping: 24 }}
      className="group relative overflow-hidden rounded-full border border-white/15 bg-white/[0.06] px-6 py-2.5 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(8,12,24,0.35)] backdrop-blur-xl transition-colors hover:border-white/30 hover:bg-white/10"
    >
      <span
        className={`absolute inset-0 bg-gradient-to-r ${glowClassName} opacity-0 transition-opacity duration-300 group-hover:opacity-100`}
      />
      <span className="relative">{label}</span>
    </motion.button>
  );
}
