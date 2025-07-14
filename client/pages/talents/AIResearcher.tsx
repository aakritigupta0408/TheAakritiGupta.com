import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

export default function AIResearcher() {
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
            <div className="w-24 h-24 bg-gradient-to-r from-cyan-500 to-cyan-700 rounded-sm flex items-center justify-center text-white text-4xl font-bold mx-auto mb-8 shadow-2xl">
              ◆
            </div>
            <h1 className="tom-ford-heading text-6xl md:text-8xl text-white mb-8">
              AI
              <br />
              <span className="gold-shimmer">RESEARCHER</span>
            </h1>
            <p className="tom-ford-subheading text-white/60 text-xl tracking-widest max-w-4xl mx-auto">
              ADVANCING THE FRONTIERS OF ARTIFICIAL INTELLIGENCE AND MACHINE
              LEARNING
            </p>
          </motion.div>
        </div>
      </section>

      {/* Research & Innovation Section */}
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
                RESEARCH
                <br />
                <span className="gold-shimmer">EXCELLENCE</span>
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
                    <div className="w-12 h-12 bg-cyan-500/20 rounded-sm flex items-center justify-center text-cyan-400 text-xl">
                      🧠
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        MACHINE LEARNING INNOVATION
                      </h3>
                      <p className="text-white/60 text-sm">
                        Novel algorithms & architectures
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Pioneering research in advanced machine learning
                    architectures, developing novel algorithms for computer
                    vision, natural language processing, and deep learning
                    optimization techniques.
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
                    <div className="w-12 h-12 bg-cyan-500/20 rounded-sm flex items-center justify-center text-cyan-400 text-xl">
                      🏆
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        YANN LECUN RECOGNITION
                      </h3>
                      <p className="text-white/60 text-sm">
                        Turing Award winner acknowledgment
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Recognized by Dr. Yann LeCun at ICLR 2019 for innovative
                    contributions to AI research, representing the highest level
                    of academic acknowledgment in the field.
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
                    <div className="w-12 h-12 bg-cyan-500/20 rounded-sm flex items-center justify-center text-cyan-400 text-xl">
                      📊
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        SCALABLE AI SYSTEMS
                      </h3>
                      <p className="text-white/60 text-sm">
                        Production-grade implementations
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Translating cutting-edge research into production systems
                    serving billions of users at Meta, eBay, and Yahoo, bridging
                    the gap between academic innovation and real-world impact.
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
                <div className="text-8xl text-cyan-400 mb-8">🔬</div>
                <h3 className="tom-ford-heading text-3xl text-white mb-6">
                  RESEARCH IMPACT
                </h3>
                <div className="grid grid-cols-2 gap-8">
                  <div className="text-center">
                    <div className="text-4xl text-cyan-400 font-bold mb-2">
                      ICLR
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      2019 RECOGNITION
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-cyan-400 font-bold mb-2">
                      10B+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      USERS IMPACTED
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-cyan-400 font-bold mb-2">
                      50+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      ML MODELS DEPLOYED
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-cyan-400 font-bold mb-2">
                      ∞
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      INNOVATION POTENTIAL
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Professional Impact Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="tom-ford-heading text-5xl text-white mb-8">
              RESEARCH
              <br />
              <span className="gold-shimmer">APPLICATIONS</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              TRANSFORMING THEORETICAL BREAKTHROUGHS INTO PRACTICAL SOLUTIONS
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
              <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-cyan-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🔍
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                COMPUTER VISION
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Advanced image recognition and analysis systems for security
                applications, including Parliament face recognition and PPE
                detection systems enhancing workplace safety.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-cyan-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                📈
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                RECOMMENDATION SYSTEMS
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                ML-driven personalization engines serving billions of users
                across e-commerce and social platforms, optimizing user
                engagement and business metrics at massive scale.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-cyan-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🚀
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                AI DEMOCRATIZATION
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Building accessible AI tools and platforms that enable
                non-technical users to leverage advanced machine learning
                capabilities for creative and business applications.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Research Philosophy Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="tom-ford-glass p-16 rounded-sm text-center"
          >
            <div className="text-6xl text-cyan-400 mb-8">🏆</div>
            <h2 className="tom-ford-heading text-4xl text-white mb-8">
              RESEARCH
              <br />
              <span className="gold-shimmer">PHILOSOPHY</span>
            </h2>
            <blockquote className="text-2xl text-white/80 font-light leading-relaxed max-w-4xl mx-auto mb-8 italic">
              "True AI research isn't just about advancing the state of the
              art—it's about building intelligence that enhances human potential
              and solves real-world problems. The best algorithms are those that
              make complex things simple."
            </blockquote>
            <div className="tom-ford-subheading text-yellow-400 text-lg tracking-widest">
              — AAKRITI GUPTA, AI RESEARCHER & INNOVATOR
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
                onClick={() => navigate("/talent/social-entrepreneur")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-teal-400/50 text-teal-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-teal-400 hover:bg-teal-400/10 transition-all duration-300"
              >
                SOCIAL ENTREPRENEUR
              </motion.button>
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
            </div>
          </motion.div>
        </div>
      </section>

      <ChatBot />
    </div>
  );
}
