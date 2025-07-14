import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

export default function Marksman() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen tom-ford-gradient relative overflow-x-hidden">
      <Navigation />

      {/* Hero Section */}
      <section className="relative z-20 pt-32 pb-20">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="text-center mb-16"
          >
            <div className="w-24 h-24 bg-gradient-to-r from-red-500 to-red-700 rounded-sm flex items-center justify-center text-white text-4xl font-bold mx-auto mb-8 shadow-2xl">
              ◎
            </div>
            <h1 className="tom-ford-heading text-6xl md:text-8xl text-white mb-8">
              PRECISION
              <br />
              <span className="gold-shimmer">MARKSMAN</span>
            </h1>
            <p className="tom-ford-subheading text-white/60 text-xl tracking-widest max-w-4xl mx-auto">
              MASTERING THE ART OF PRECISION, FOCUS, AND TACTICAL EXCELLENCE
            </p>
          </motion.div>
        </div>
      </section>

      {/* Skills & Training Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid lg:grid-cols-2 gap-16 items-center"
          >
            <div>
              <h2 className="tom-ford-heading text-4xl text-white mb-8">
                TACTICAL
                <br />
                <span className="gold-shimmer">EXPERTISE</span>
              </h2>
              <div className="space-y-8">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.1 }}
                  className="tom-ford-card p-6 rounded-sm"
                >
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-12 h-12 bg-red-500/20 rounded-sm flex items-center justify-center text-red-400 text-xl">
                      🎯
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        PRECISION SHOOTING
                      </h3>
                      <p className="text-white/60 text-sm">
                        Target accuracy & consistency
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Extensive training in precision shooting techniques,
                    developing exceptional hand-eye coordination and mental
                    focus that translates directly to professional excellence
                    and attention to detail.
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.2 }}
                  className="tom-ford-card p-6 rounded-sm"
                >
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-12 h-12 bg-red-500/20 rounded-sm flex items-center justify-center text-red-400 text-xl">
                      🧠
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        MENTAL DISCIPLINE
                      </h3>
                      <p className="text-white/60 text-sm">
                        Focus & concentration mastery
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Rigorous mental training for sustained concentration,
                    emotional control under pressure, and split-second decision
                    making - skills that enhance problem-solving in complex AI
                    systems.
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.3 }}
                  className="tom-ford-card p-6 rounded-sm"
                >
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-12 h-12 bg-red-500/20 rounded-sm flex items-center justify-center text-red-400 text-xl">
                      ⚡
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        TACTICAL AWARENESS
                      </h3>
                      <p className="text-white/60 text-sm">
                        Strategic thinking & positioning
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Advanced tactical training emphasizing situational
                    awareness, strategic positioning, and risk assessment -
                    directly applicable to system architecture and security
                    considerations.
                  </p>
                </motion.div>
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="tom-ford-glass p-12 rounded-sm text-center">
                <div className="text-8xl text-red-400 mb-8">🏹</div>
                <h3 className="tom-ford-heading text-3xl text-white mb-6">
                  PRECISION METRICS
                </h3>
                <div className="grid grid-cols-2 gap-8">
                  <div className="text-center">
                    <div className="text-4xl text-red-400 font-bold mb-2">
                      95%
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      ACCURACY RATE
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-red-400 font-bold mb-2">
                      5+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      YEARS TRAINING
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-red-400 font-bold mb-2">
                      100m
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      EFFECTIVE RANGE
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-red-400 font-bold mb-2">
                      ∞
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      FOCUS DISCIPLINE
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Professional Connection Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="tom-ford-heading text-5xl text-white mb-8">
              PROFESSIONAL
              <br />
              <span className="gold-shimmer">TRANSLATION</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              HOW MARKSMANSHIP EXCELLENCE ENHANCES TECHNICAL LEADERSHIP
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-red-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🎯
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                PRECISION IN CODE
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                The same precision required for accurate shooting translates to
                writing clean, error-free code and implementing exact
                algorithmic solutions in machine learning models.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-red-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🧠
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                FOCUS UNDER PRESSURE
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Mental discipline developed through marksmanship enables
                sustained concentration during complex debugging sessions and
                high-stakes system deployments.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-red-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                ⚡
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                STRATEGIC THINKING
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Tactical awareness and positioning skills enhance system
                architecture decisions and strategic technical planning in
                large-scale engineering projects.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Training Philosophy Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="tom-ford-glass p-16 rounded-sm text-center"
          >
            <div className="text-6xl text-red-400 mb-8">🏆</div>
            <h2 className="tom-ford-heading text-4xl text-white mb-8">
              EXCELLENCE
              <br />
              <span className="gold-shimmer">PHILOSOPHY</span>
            </h2>
            <blockquote className="text-2xl text-white/80 font-light leading-relaxed max-w-4xl mx-auto mb-8 italic">
              "Precision is not about perfection—it's about consistency,
              discipline, and the relentless pursuit of improvement. Every shot
              teaches you something, just like every line of code."
            </blockquote>
            <div className="tom-ford-subheading text-yellow-400 text-lg tracking-widest">
              — AAKRITI GUPTA, PRECISION MARKSMAN & ML ENGINEER
            </div>
          </motion.div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="tom-ford-heading text-3xl text-white mb-8">
              EXPLORE MORE
              <br />
              <span className="gold-shimmer">TALENTS</span>
            </h2>
            <div className="flex justify-center gap-6 flex-wrap">
              <motion.button
                onClick={() => navigate("/talent/equestrian")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-amber-400/50 text-amber-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-amber-400 hover:bg-amber-400/10 transition-all duration-300"
              >
                EQUESTRIAN
              </motion.button>
              <motion.button
                onClick={() => navigate("/talent/aviator")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-blue-400/50 text-blue-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-blue-400 hover:bg-blue-400/10 transition-all duration-300"
              >
                AVIATOR
              </motion.button>
              <motion.button
                onClick={() => navigate("/talent/motorcyclist")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-purple-400/50 text-purple-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-purple-400 hover:bg-purple-400/10 transition-all duration-300"
              >
                MOTORCYCLIST
              </motion.button>
              <motion.button
                onClick={() => navigate("/talent/pianist")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-green-400/50 text-green-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-green-400 hover:bg-green-400/10 transition-all duration-300"
              >
                PIANIST
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>

      <ChatBot />
    </div>
  );
}
