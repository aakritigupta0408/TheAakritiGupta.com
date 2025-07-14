import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

export default function Motorcyclist() {
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
            <div className="w-24 h-24 bg-gradient-to-r from-purple-500 to-purple-700 rounded-sm flex items-center justify-center text-white text-4xl font-bold mx-auto mb-8 shadow-2xl">
              ◐
            </div>
            <h1 className="tom-ford-heading text-6xl md:text-8xl text-white mb-8">
              SPEED &
              <br />
              <span className="gold-shimmer">FREEDOM</span>
            </h1>
            <p className="tom-ford-subheading text-white/60 text-xl tracking-widest max-w-4xl mx-auto">
              MASTERING HIGH-PERFORMANCE RIDING AND MECHANICAL PRECISION
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
                PERFORMANCE
                <br />
                <span className="gold-shimmer">MASTERY</span>
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
                    <div className="w-12 h-12 bg-purple-500/20 rounded-sm flex items-center justify-center text-purple-400 text-xl">
                      🏍️
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        HIGH-SPEED CONTROL
                      </h3>
                      <p className="text-white/60 text-sm">
                        Precision at extreme speeds
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Maintaining absolute control while operating
                    high-performance machinery at speed develops exceptional
                    reaction times and the ability to handle complex systems
                    under pressure.
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
                    <div className="w-12 h-12 bg-purple-500/20 rounded-sm flex items-center justify-center text-purple-400 text-xl">
                      🔧
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        MECHANICAL EXPERTISE
                      </h3>
                      <p className="text-white/60 text-sm">
                        Engine tuning & optimization
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Deep understanding of mechanical systems, performance
                    tuning, and optimization translates directly to system
                    performance analysis and infrastructure optimization.
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
                    <div className="w-12 h-12 bg-purple-500/20 rounded-sm flex items-center justify-center text-purple-400 text-xl">
                      ⚡
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        RISK MANAGEMENT
                      </h3>
                      <p className="text-white/60 text-sm">
                        Calculated risks & safety
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Balancing performance with safety requires sophisticated
                    risk assessment and mitigation strategies - essential for
                    production system deployments and architectural decisions.
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
                <div className="text-8xl text-purple-400 mb-8">🏁</div>
                <h3 className="tom-ford-heading text-3xl text-white mb-6">
                  RIDING ACHIEVEMENTS
                </h3>
                <div className="grid grid-cols-2 gap-8">
                  <div className="text-center">
                    <div className="text-4xl text-purple-400 font-bold mb-2">
                      10+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      YEARS RIDING
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-purple-400 font-bold mb-2">
                      200+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      MPH TOP SPEED
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-purple-400 font-bold mb-2">
                      6
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      BIKE TYPES MASTERED
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-purple-400 font-bold mb-2">
                      ∞
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      FREEDOM & SPEED
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
              PERFORMANCE
              <br />
              <span className="gold-shimmer">OPTIMIZATION</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              HOW MOTORCYCLE EXPERTISE ENHANCES SYSTEM PERFORMANCE
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
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-purple-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                ⚡
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                SPEED OPTIMIZATION
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Understanding performance tuning in mechanical systems
                translates to optimizing ML model inference times and database
                query performance for maximum efficiency.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-purple-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🔧
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                SYSTEM MECHANICS
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Deep mechanical understanding helps diagnose complex system
                issues, optimize infrastructure performance, and design robust
                distributed architectures.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-purple-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🎯
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                CALCULATED RISKS
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Balancing performance gains with system stability mirrors
                production deployment strategies and technical debt management
                in high-scale systems.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Philosophy Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="tom-ford-glass p-16 rounded-sm text-center"
          >
            <div className="text-6xl text-purple-400 mb-8">🏆</div>
            <h2 className="tom-ford-heading text-4xl text-white mb-8">
              PERFORMANCE
              <br />
              <span className="gold-shimmer">PHILOSOPHY</span>
            </h2>
            <blockquote className="text-2xl text-white/80 font-light leading-relaxed max-w-4xl mx-auto mb-8 italic">
              "True performance isn't just about speed—it's about the perfect
              balance of power, control, and precision. Every system, whether
              mechanical or digital, has an optimal configuration waiting to be
              discovered."
            </blockquote>
            <div className="tom-ford-subheading text-yellow-400 text-lg tracking-widest">
              — AAKRITI GUPTA, MOTORCYCLIST & PERFORMANCE ENGINEER
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
                onClick={() => navigate("/talent/marksman")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-red-400/50 text-red-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-red-400 hover:bg-red-400/10 transition-all duration-300"
              >
                MARKSMAN
              </motion.button>
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
