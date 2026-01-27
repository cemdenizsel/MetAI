'use client';

import { Heart, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import type { MentalHealthAnalysis, MentalHealthAssessment } from '@/types/emotion';
import { cn } from '@/components/ui/utils';

interface MentalHealthCardProps {
  assessment: MentalHealthAnalysis | MentalHealthAssessment;
}

const statusConfig = {
  excellent: {
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-100 dark:bg-green-900/30',
    borderColor: 'border-green-200 dark:border-green-800',
    icon: CheckCircle,
    label: 'Excellent',
  },
  good: {
    color: 'text-teal-600 dark:text-teal-400',
    bgColor: 'bg-teal-100 dark:bg-teal-900/30',
    borderColor: 'border-teal-200 dark:border-teal-800',
    icon: CheckCircle,
    label: 'Good',
  },
  moderate: {
    color: 'text-amber-600 dark:text-amber-400',
    bgColor: 'bg-amber-100 dark:bg-amber-900/30',
    borderColor: 'border-amber-200 dark:border-amber-800',
    icon: Info,
    label: 'Moderate',
  },
  concerning: {
    color: 'text-orange-600 dark:text-orange-400',
    bgColor: 'bg-orange-100 dark:bg-orange-900/30',
    borderColor: 'border-orange-200 dark:border-orange-800',
    icon: AlertTriangle,
    label: 'Concerning',
  },
  critical: {
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-100 dark:bg-red-900/30',
    borderColor: 'border-red-200 dark:border-red-800',
    icon: AlertTriangle,
    label: 'Critical',
  },
};

export function MentalHealthCard({ assessment }: MentalHealthCardProps) {
  // Handle both MentalHealthAnalysis and MentalHealthAssessment types
  const status = 'status' in assessment ? assessment.status : 'moderate';
  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.moderate;
  const StatusIcon = config.icon;

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400';
    if (score >= 60) return 'text-teal-600 dark:text-teal-400';
    if (score >= 40) return 'text-amber-600 dark:text-amber-400';
    if (score >= 20) return 'text-orange-600 dark:text-orange-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getProgressColor = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-teal-500';
    if (score >= 40) return 'bg-amber-500';
    if (score >= 20) return 'bg-orange-500';
    return 'bg-red-500';
  };

  return (
    <div
      className={cn(
        'p-6 rounded-xl border',
        config.bgColor,
        config.borderColor
      )}
    >
      <div className="flex items-center gap-3 mb-4">
        <div
          className={cn(
            'w-10 h-10 rounded-full flex items-center justify-center',
            config.bgColor
          )}
        >
          <Heart className={cn('w-5 h-5', config.color)} />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Mental Health Assessment
          </h3>
          <div className="flex items-center gap-2">
            <StatusIcon className={cn('w-4 h-4', config.color)} />
            <span className={cn('text-sm font-medium', config.color)}>
              {config.label}
            </span>
          </div>
        </div>
      </div>

      {/* Score */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Wellness Score
          </span>
          <span
            className={cn(
              'text-2xl font-bold',
              getScoreColor('mental_health_score' in assessment ? assessment.mental_health_score : assessment.score)
            )}
          >
            {('mental_health_score' in assessment ? assessment.mental_health_score : assessment.score)}/100
          </span>
        </div>
        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full transition-all duration-500',
              getProgressColor(assessment.score)
            )}
            style={{
              width: `${('mental_health_score' in assessment ? assessment.mental_health_score : assessment.score)}%`,
            }}
          />
        </div>
      </div>

      {/* Recommendation */}
      {('recommendation' in assessment ? assessment.recommendation : null) && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Recommendation
          </h4>
          <p className="text-gray-600 dark:text-gray-400">
            {('recommendation' in assessment ? assessment.recommendation : '')}
          </p>
        </div>
      )}

      {/* Factors */}
      {('factors' in assessment ? assessment.factors : null) &&
        ('factors' in assessment ? assessment.factors : []).length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Contributing Factors
            </h4>
            <ul className="space-y-1">
              {('factors' in assessment ? assessment.factors : []).map((factor, index) => (
                <li
                  key={index}
                  className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-gray-400" />
                  {factor}
                </li>
              ))}
            </ul>
          </div>
        )}
    </div>
  );
}
