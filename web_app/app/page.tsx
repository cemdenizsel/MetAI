'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useTheme } from 'next-themes';
import {
  Upload,
  Brain,
  BarChart3,
  Heart,
  Shield,
  Zap,
  Sun,
  Moon,
  ArrowRight,
  CheckCircle,
  Sparkles,
  Video,
  Activity,
} from 'lucide-react';

export default function LandingPage() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-teal-600 to-teal-700 rounded-lg flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900 dark:text-white">
              Emotion Analysis
            </span>
          </div>

          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5 text-gray-600" />
              ) : (
                <Sun className="w-5 h-5 text-gray-400" />
              )}
            </button>
            <Link
              href="/login"
              className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:text-teal-600 dark:hover:text-teal-400 font-medium transition-colors"
            >
              Sign In
            </Link>
            <Link
              href="/register"
              className="px-6 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg font-medium transition-colors"
            >
              Get Started
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 pt-20 pb-24">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-300 rounded-full text-sm font-medium mb-6 border border-teal-200 dark:border-teal-800">
              <Zap className="w-4 h-4" />
              <span>AI-Powered Emotion Recognition</span>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6 leading-tight">
              Understand emotions from video content
            </h1>

            <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
              Advanced AI analysis to detect and understand emotional patterns
              in video. Get mental health insights, track emotional trajectories,
              and receive personalized recommendations.
            </p>

            {/* Main CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 mb-8">
              <Link
                href="/register"
                className="group flex items-center justify-center gap-2 px-6 py-3.5 bg-teal-600 hover:bg-teal-700 text-white rounded-lg font-semibold text-base shadow-lg shadow-teal-600/20 transition-all"
              >
                <span>Start Free Analysis</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>

              <Link
                href="/login"
                className="flex items-center justify-center gap-2 px-6 py-3.5 bg-white dark:bg-gray-800 border-2 border-gray-300 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 text-gray-900 dark:text-white rounded-lg font-semibold text-base transition-all"
              >
                <Video className="w-5 h-5" />
                <span>Watch Demo</span>
              </Link>
            </div>

            {/* Features List */}
            <div className="flex flex-wrap items-center gap-6 text-sm text-gray-500 dark:text-gray-500">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-teal-600" />
                <span>7 Emotion Categories</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-teal-600" />
                <span>Mental Health Insights</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-teal-600" />
                <span>AI-Powered Recommendations</span>
              </div>
            </div>
          </div>

          {/* Hero Image */}
          <div className="relative">
            <div className="relative rounded-2xl overflow-hidden shadow-2xl border border-gray-200 dark:border-gray-800">
              <div className="aspect-video bg-gradient-to-br from-teal-100 to-teal-200 dark:from-teal-900 dark:to-teal-800 flex items-center justify-center">
                <div className="text-center p-8">
                  <div className="w-24 h-24 mx-auto mb-4 bg-white dark:bg-gray-800 rounded-full flex items-center justify-center shadow-lg">
                    <Brain className="w-12 h-12 text-teal-600" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                    Emotion Analysis
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Upload a video to get started
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-gray-50 dark:bg-gray-800/50 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-teal-600 dark:text-teal-400 mb-2">
                7
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                Emotion Categories
              </div>
            </div>
            <div>
              <div className="text-4xl font-bold text-teal-600 dark:text-teal-400 mb-2">
                98%
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                Accuracy Rate
              </div>
            </div>
            <div>
              <div className="text-4xl font-bold text-teal-600 dark:text-teal-400 mb-2">
                Real-time
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                Processing
              </div>
            </div>
            <div>
              <div className="text-4xl font-bold text-teal-600 dark:text-teal-400 mb-2">
                24/7
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                Availability
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Powerful emotion analysis features
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Everything you need to understand emotional content in video
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 transition-all hover:shadow-lg">
            <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center mb-4">
              <Upload className="w-6 h-6 text-teal-600 dark:text-teal-400" />
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
              Easy Video Upload
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Drag and drop video files or upload from your device. Supports
              MP4, AVI, MOV, WebM, MKV, and FLV formats.
            </p>
          </div>

          <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 transition-all hover:shadow-lg">
            <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center mb-4">
              <Brain className="w-6 h-6 text-teal-600 dark:text-teal-400" />
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
              Multi-Model Analysis
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Leverages multiple AI models for accurate emotion detection across
              7 categories: happy, sad, angry, fear, surprise, disgust, neutral.
            </p>
          </div>

          <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 transition-all hover:shadow-lg">
            <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center mb-4">
              <BarChart3 className="w-6 h-6 text-teal-600 dark:text-teal-400" />
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
              Visual Analytics
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Interactive charts showing emotion distribution over time. Track
              emotional trajectories and identify patterns.
            </p>
          </div>

          <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 transition-all hover:shadow-lg">
            <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center mb-4">
              <Heart className="w-6 h-6 text-teal-600 dark:text-teal-400" />
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
              Mental Health Insights
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Get mental health scores and personalized recommendations based on
              detected emotional patterns.
            </p>
          </div>

          <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 transition-all hover:shadow-lg">
            <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-teal-600 dark:text-teal-400" />
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
              Privacy First
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Your videos are processed securely and can be deleted after
              analysis. Choose between cloud or local AI processing.
            </p>
          </div>

          <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 transition-all hover:shadow-lg">
            <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center mb-4">
              <Activity className="w-6 h-6 text-teal-600 dark:text-teal-400" />
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
              Batch Processing
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Process multiple videos simultaneously with async job tracking.
              View progress and results in real-time.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="bg-gray-50 dark:bg-gray-800/50 py-20">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              How it works
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400">
              Get emotion insights in three simple steps
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-12">
            {/* Step 1 */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-700">
              <div className="relative h-48 bg-gradient-to-br from-teal-100 to-teal-200 dark:from-teal-900 dark:to-teal-800 flex items-center justify-center">
                <Upload className="w-16 h-16 text-teal-600 dark:text-teal-400" />
                <div className="absolute top-4 left-4 w-12 h-12 bg-teal-600 text-white rounded-full flex items-center justify-center text-xl font-bold shadow-lg">
                  1
                </div>
              </div>
              <div className="p-6">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  Upload Your Video
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Select or drag-and-drop a video file. We support most common
                  video formats up to 500MB.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-700">
              <div className="relative h-48 bg-gradient-to-br from-teal-100 to-teal-200 dark:from-teal-900 dark:to-teal-800 flex items-center justify-center">
                <Brain className="w-16 h-16 text-teal-600 dark:text-teal-400" />
                <div className="absolute top-4 left-4 w-12 h-12 bg-teal-600 text-white rounded-full flex items-center justify-center text-xl font-bold shadow-lg">
                  2
                </div>
              </div>
              <div className="p-6">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  AI Analyzes Emotions
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Our AI processes each frame, detecting facial expressions and
                  classifying emotions with high accuracy.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-700">
              <div className="relative h-48 bg-gradient-to-br from-teal-100 to-teal-200 dark:from-teal-900 dark:to-teal-800 flex items-center justify-center">
                <BarChart3 className="w-16 h-16 text-teal-600 dark:text-teal-400" />
                <div className="absolute top-4 left-4 w-12 h-12 bg-teal-600 text-white rounded-full flex items-center justify-center text-xl font-bold shadow-lg">
                  3
                </div>
              </div>
              <div className="p-6">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  Get Detailed Insights
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  View emotion distribution, temporal analysis, mental health
                  assessment, and AI-generated recommendations.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <div className="bg-gradient-to-br from-teal-600 to-teal-700 rounded-3xl p-12 md:p-16 text-center text-white">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Ready to analyze emotions?
          </h2>
          <p className="text-xl mb-10 text-teal-100 max-w-2xl mx-auto">
            Start your free analysis today and discover emotional insights in
            your video content
          </p>

          <Link
            href="/register"
            className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white text-teal-700 rounded-xl font-semibold text-lg hover:bg-gray-100 transition-colors shadow-lg"
          >
            <Sparkles className="w-5 h-5" />
            <span>Get Started Free</span>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-6 py-12 border-t border-gray-200 dark:border-gray-700">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-teal-500 to-teal-600 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-gray-900 dark:text-white">
              Emotion Analysis
            </span>
          </div>

          <div className="text-sm text-gray-600 dark:text-gray-400">
            © 2026 Emotion Analysis App. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
