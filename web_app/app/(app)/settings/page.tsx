'use client';

import { useState } from 'react';
import { useTheme } from 'next-themes';
import {
  Settings,
  User,
  Palette,
  Brain,
  Cloud,
  Server,
  Sun,
  Moon,
  Monitor,
  Save,
  Check,
} from 'lucide-react';
import { useAuthStore } from '@/lib/stores/authStore';

type LLMProvider = 'cloud' | 'local';

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const { user } = useAuthStore();
  const [defaultProvider, setDefaultProvider] = useState<LLMProvider>('cloud');
  const [includeAiAnalysis, setIncludeAiAnalysis] = useState(true);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    localStorage.setItem(
      'emotion_settings',
      JSON.stringify({
        defaultProvider,
        includeAiAnalysis,
      })
    );
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-3xl space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Settings
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Manage your account and analysis preferences
        </p>
      </div>

      {/* Account Section */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <User className="w-5 h-5 text-teal-600" />
            Account
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Email
            </label>
            <p className="text-gray-900 dark:text-white">{user?.email || '-'}</p>
          </div>
          {user?.name && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Name
              </label>
              <p className="text-gray-900 dark:text-white">{user.name}</p>
            </div>
          )}
          {user?.created_at && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Member Since
              </label>
              <p className="text-gray-900 dark:text-white">
                {new Date(user.created_at).toLocaleDateString()}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Appearance Section */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <Palette className="w-5 h-5 text-teal-600" />
            Appearance
          </h2>
        </div>
        <div className="p-6">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Theme
          </label>
          <div className="grid grid-cols-3 gap-3">
            <button
              onClick={() => setTheme('light')}
              className={`flex flex-col items-center gap-2 p-4 rounded-lg border transition-colors ${
                theme === 'light'
                  ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-teal-400'
              }`}
            >
              <Sun
                className={`w-6 h-6 ${
                  theme === 'light'
                    ? 'text-teal-600'
                    : 'text-gray-400'
                }`}
              />
              <span
                className={`text-sm font-medium ${
                  theme === 'light'
                    ? 'text-teal-700 dark:text-teal-400'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
              >
                Light
              </span>
            </button>
            <button
              onClick={() => setTheme('dark')}
              className={`flex flex-col items-center gap-2 p-4 rounded-lg border transition-colors ${
                theme === 'dark'
                  ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-teal-400'
              }`}
            >
              <Moon
                className={`w-6 h-6 ${
                  theme === 'dark'
                    ? 'text-teal-600'
                    : 'text-gray-400'
                }`}
              />
              <span
                className={`text-sm font-medium ${
                  theme === 'dark'
                    ? 'text-teal-700 dark:text-teal-400'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
              >
                Dark
              </span>
            </button>
            <button
              onClick={() => setTheme('system')}
              className={`flex flex-col items-center gap-2 p-4 rounded-lg border transition-colors ${
                theme === 'system'
                  ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-teal-400'
              }`}
            >
              <Monitor
                className={`w-6 h-6 ${
                  theme === 'system'
                    ? 'text-teal-600'
                    : 'text-gray-400'
                }`}
              />
              <span
                className={`text-sm font-medium ${
                  theme === 'system'
                    ? 'text-teal-700 dark:text-teal-400'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
              >
                System
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Analysis Defaults Section */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <Brain className="w-5 h-5 text-teal-600" />
            Analysis Defaults
          </h2>
        </div>
        <div className="p-6 space-y-6">
          {/* Default AI Provider */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
              Default AI Provider
            </label>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setDefaultProvider('cloud')}
                className={`flex items-center justify-center gap-2 p-4 rounded-lg border transition-colors ${
                  defaultProvider === 'cloud'
                    ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-teal-400'
                }`}
              >
                <Cloud
                  className={`w-5 h-5 ${
                    defaultProvider === 'cloud'
                      ? 'text-teal-600'
                      : 'text-gray-400'
                  }`}
                />
                <span
                  className={`font-medium ${
                    defaultProvider === 'cloud'
                      ? 'text-teal-700 dark:text-teal-400'
                      : 'text-gray-600 dark:text-gray-400'
                  }`}
                >
                  Cloud AI
                </span>
              </button>
              <button
                onClick={() => setDefaultProvider('local')}
                className={`flex items-center justify-center gap-2 p-4 rounded-lg border transition-colors ${
                  defaultProvider === 'local'
                    ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-teal-400'
                }`}
              >
                <Server
                  className={`w-5 h-5 ${
                    defaultProvider === 'local'
                      ? 'text-teal-600'
                      : 'text-gray-400'
                  }`}
                />
                <span
                  className={`font-medium ${
                    defaultProvider === 'local'
                      ? 'text-teal-700 dark:text-teal-400'
                      : 'text-gray-600 dark:text-gray-400'
                  }`}
                >
                  Local AI
                </span>
              </button>
            </div>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              Cloud AI provides faster processing. Local AI keeps data on your
              machine.
            </p>
          </div>

          {/* Include AI Analysis */}
          <div className="flex items-center justify-between">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Include AI Analysis by Default
              </label>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Automatically include detailed insights and recommendations
              </p>
            </div>
            <button
              onClick={() => setIncludeAiAnalysis(!includeAiAnalysis)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                includeAiAnalysis
                  ? 'bg-teal-500'
                  : 'bg-gray-300 dark:bg-gray-600'
              }`}
            >
              <span
                className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                  includeAiAnalysis ? 'translate-x-7' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={saved}
          className="flex items-center gap-2 px-6 py-3 bg-teal-600 hover:bg-teal-700 text-white rounded-lg font-semibold transition-colors disabled:opacity-50"
        >
          {saved ? (
            <>
              <Check className="w-5 h-5" />
              Saved!
            </>
          ) : (
            <>
              <Save className="w-5 h-5" />
              Save Settings
            </>
          )}
        </button>
      </div>
    </div>
  );
}
