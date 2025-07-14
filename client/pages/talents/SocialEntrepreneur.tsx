import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ChatBot from "@/components/ChatBot";

export default function SocialEntrepreneur() {
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
            <div className="w-24 h-24 bg-gradient-to-r from-teal-500 to-teal-700 rounded-sm flex items-center justify-center text-white text-4xl font-bold mx-auto mb-8 shadow-2xl">
              ◇
            </div>
            <h1 className="tom-ford-heading text-6xl md:text-8xl text-white mb-8">
              SOCIAL
              <br />
              <span className="gold-shimmer">ENTREPRENEUR</span>
            </h1>
            <p className="tom-ford-subheading text-white/60 text-xl tracking-widest max-w-4xl mx-auto">
              BUILDING TECHNOLOGY FOR POSITIVE IMPACT AND SOCIAL TRANSFORMATION
            </p>
          </motion.div>
        </div>
      </section>

      {/* Vision & Impact Section */}
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
                SOCIAL
                <br />
                <span className="gold-shimmer">INNOVATION</span>
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
                    <div className="w-12 h-12 bg-teal-500/20 rounded-sm flex items-center justify-center text-teal-400 text-xl">
                      💎
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        SWARNAWASTRA VISION
                      </h3>
                      <p className="text-white/60 text-sm">
                        Democratizing luxury through AI
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Founded Swarnawastra to make luxury fashion accessible
                    through AI-driven design and lab-grown diamonds, bridging
                    the gap between high-end aesthetics and affordability for
                    emerging markets.
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
                    <div className="w-12 h-12 bg-teal-500/20 rounded-sm flex items-center justify-center text-teal-400 text-xl">
                      🌍
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        GLOBAL ACCESSIBILITY
                      </h3>
                      <p className="text-white/60 text-sm">
                        Technology for underserved markets
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Building AI systems that make sophisticated technology
                    accessible to underserved communities, focusing on
                    educational tools, healthcare applications, and economic
                    empowerment through innovation.
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
                    <div className="w-12 h-12 bg-teal-500/20 rounded-sm flex items-center justify-center text-teal-400 text-xl">
                      🚀
                    </div>
                    <div>
                      <h3 className="tom-ford-subheading text-white text-lg tracking-wider">
                        SUSTAINABLE INNOVATION
                      </h3>
                      <p className="text-white/60 text-sm">
                        Ethical technology development
                      </p>
                    </div>
                  </div>
                  <p className="text-white/70 font-light leading-relaxed">
                    Championing sustainable technology practices, ethical AI
                    development, and creating business models that prioritize
                    social impact alongside financial success and environmental
                    responsibility.
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
                <div className="text-8xl text-teal-400 mb-8">🌟</div>
                <h3 className="tom-ford-heading text-3xl text-white mb-6">
                  IMPACT METRICS
                </h3>
                <div className="grid grid-cols-2 gap-8">
                  <div className="text-center">
                    <div className="text-4xl text-teal-400 font-bold mb-2">
                      3
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      COMPANIES FOUNDED
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-teal-400 font-bold mb-2">
                      1M+
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      LIVES IMPACTED
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-teal-400 font-bold mb-2">
                      15
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      SOCIAL INITIATIVES
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-4xl text-teal-400 font-bold mb-2">
                      ∞
                    </div>
                    <div className="tom-ford-subheading text-white/60 text-sm tracking-wider">
                      POTENTIAL FOR GOOD
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
              ENTREPRENEURIAL
              <br />
              <span className="gold-shimmer">VENTURES</span>
            </h2>
            <p className="tom-ford-subheading text-white/60 text-lg tracking-widest max-w-4xl mx-auto">
              TRANSFORMING IDEAS INTO SCALABLE SOLUTIONS FOR SOCIAL GOOD
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
              <div className="w-16 h-16 bg-gradient-to-r from-teal-500 to-teal-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                💎
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                LUXURY DEMOCRATIZATION
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Swarnawastra leverages AI and lab-grown materials to make luxury
                fashion accessible, creating economic opportunities while
                maintaining environmental sustainability and ethical practices.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-teal-500 to-teal-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                🏥
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                HEALTHCARE INNOVATION
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Developing AI-powered healthcare solutions for underserved
                communities, including diagnostic tools and telemedicine
                platforms that bring quality care to remote areas.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="tom-ford-card p-8 rounded-sm text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-teal-500 to-teal-700 rounded-sm flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                📚
              </div>
              <h3 className="tom-ford-subheading text-white text-lg tracking-wider mb-4">
                EDUCATION EMPOWERMENT
              </h3>
              <p className="text-white/70 font-light leading-relaxed">
                Creating accessible AI education platforms and tools that
                democratize learning, enabling students from diverse backgrounds
                to access world-class technical education.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Leadership Philosophy Section */}
      <section className="relative z-20 py-20 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="tom-ford-glass p-16 rounded-sm text-center"
          >
            <div className="text-6xl text-teal-400 mb-8">🏆</div>
            <h2 className="tom-ford-heading text-4xl text-white mb-8">
              ENTREPRENEURIAL
              <br />
              <span className="gold-shimmer">PHILOSOPHY</span>
            </h2>
            <blockquote className="text-2xl text-white/80 font-light leading-relaxed max-w-4xl mx-auto mb-8 italic">
              "True entrepreneurship isn't just about building successful
              companies—it's about creating technology that empowers people,
              solves meaningful problems, and makes the world more equitable.
              Profit should be a byproduct of purpose, not the purpose itself."
            </blockquote>
            <div className="tom-ford-subheading text-yellow-400 text-lg tracking-widest">
              — AAKRITI GUPTA, SOCIAL ENTREPRENEUR & IMPACT LEADER
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
                onClick={() => navigate("/talent/ai-researcher")}
                whileHover={{ scale: 1.02, y: -2 }}
                className="px-6 py-3 border border-cyan-400/50 text-cyan-400 rounded-sm tom-ford-subheading text-sm tracking-wider hover:border-cyan-400 hover:bg-cyan-400/10 transition-all duration-300"
              >
                AI RESEARCHER
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
