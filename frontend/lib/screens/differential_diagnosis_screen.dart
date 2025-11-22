import 'package:flutter/material.dart';

import '../services/api_client.dart';
import '../widgets/modality_radar_chart.dart';
import '../widgets/vitals_sparkline.dart';

class DifferentialDiagnosisScreen extends StatefulWidget {
  const DifferentialDiagnosisScreen({super.key});

  static const routeName = '/ddx';

  @override
  State<DifferentialDiagnosisScreen> createState() => _DifferentialDiagnosisScreenState();
}

class _DifferentialDiagnosisScreenState extends State<DifferentialDiagnosisScreen> {
  String? _caseId;
  Map<String, dynamic>? _analysis;
  bool _loading = false;
  String? _error;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _caseId ??= ModalRoute.of(context)?.settings.arguments as String?;
    if (_caseId != null && _analysis == null && !_loading) {
      _fetchAnalysis();
    }
  }

  Future<void> _fetchAnalysis() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = ApiClient();
      final data = await api.getAnalysis(_caseId!);
      if (!mounted) return;
      setState(() {
        _analysis = data;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load analysis: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final fusion = _analysis?['fusion'] as Map<String, dynamic>?;
    final triage = fusion?['triage'] as String? ?? 'High Severity';
    final conditions = (fusion?['conditions'] as List?) ?? [];
    final primary = conditions.isNotEmpty ? conditions.first as Map<String, dynamic> : null;
    final primaryName = primary?['condition'] as String? ?? 'Acute Myocardial Infarction';
    final primaryProb = (primary?['prob'] as num?)?.toDouble() ?? 0.85;

    final nlp = _analysis?['nlp'];
    final labs = _analysis?['labs'];
    final imaging = _analysis?['imaging'];
    final vitals = _analysis?['vitals'] as Map<String, dynamic>?;

    // Prefer backend-provided modality_scores if present, else fall back to presence flags.
    final modalityScores = fusion?['modality_scores'] as Map<String, dynamic>?;
    final nlpScore = (modalityScores?['nlp'] as num?)?.toDouble() ?? (nlp == null ? 0.0 : 1.0);
    final imagingScore =
        (modalityScores?['imaging'] as num?)?.toDouble() ?? (imaging == null ? 0.0 : 1.0);
    final labScore = (modalityScores?['labs'] as num?)?.toDouble() ?? (labs == null ? 0.0 : 1.0);
    final vitalsScore =
        (modalityScores?['vitals'] as num?)?.toDouble() ?? (vitals == null ? 0.0 : 1.0);

    // Use real vitals HR/SpO2 time-series if provided by the backend.
    final List<double> hrSeries;
    final List<double> spo2Series;
    if (vitals != null) {
      final hr = (vitals['heart_rate'] as List?) ?? const [];
      final s2 = (vitals['spo2'] as List?) ?? const [];
      hrSeries = hr.map((e) => (e as num).toDouble()).toList();
      spo2Series = s2.map((e) => (e as num).toDouble()).toList();
    } else {
      hrSeries = const [];
      spo2Series = const [];
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Differential Diagnosis & Insights'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (_caseId != null)
            Text(
              'Case: $_caseId',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          Text(
            'Patient Summary',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 4),
          Text(
            'CRITICAL ALERT: Acute Chest Pain - Vitals Stable',
            style: Theme.of(context)
                .textTheme
                .bodySmall
                ?.copyWith(color: Colors.redAccent),
          ),
          const SizedBox(height: 16),
          if (_loading) const LinearProgressIndicator(),
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(_error!, style: const TextStyle(color: Colors.red)),
          ],
          const SizedBox(height: 8),
          if (_analysis != null) ...[
            Text('Modality Contributions',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            ModalityRadarChart(
              nlpScore: nlpScore,
              imagingScore: imagingScore,
              labScore: labScore,
              vitalsScore: vitalsScore,
            ),
            const SizedBox(height: 16),
            if (hrSeries.isNotEmpty || spo2Series.isNotEmpty) ...[
              Text('Vitals Trend',
                  style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              VitalsSparkline(
                heartRate: hrSeries,
                spo2: spo2Series,
              ),
              const SizedBox(height: 16),
            ],
          ],
          ExpansionTile(
            title: Text('1. $primaryName'),
            subtitle: Text('$triage  •  ${(primaryProb * 100).round()}% Probability'),
            childrenPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            children: const [
              _InsightSection(),
              SizedBox(height: 16),
              _UrgentChecklist(),
            ],
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _caseId == null
                  ? null
                  : () {
                      Navigator.pushNamed(
                        context,
                        '/imaging',
                        arguments: _caseId,
                      );
                    },
              child: const Text('Open Imaging Review'),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: _caseId == null
                  ? null
                  : () {
                      Navigator.pushNamed(
                        context,
                        '/agent-timeline',
                        arguments: _caseId,
                      );
                    },
              child: const Text('View Agent & Posterior Timeline'),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: _caseId == null
                  ? null
                  : () {
                      Navigator.pushNamed(
                        context,
                        '/report',
                        arguments: _caseId,
                      );
                    },
              child: const Text('Generate Clinical Report'),
            ),
          ),
        ],
      ),
    );
  }
}

class _InsightSection extends StatelessWidget {
  const _InsightSection();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Insights', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        const Text('- Key findings pointing towards this diagnosis...'),
      ],
    );
  }
}

class _UrgentChecklist extends StatelessWidget {
  const _UrgentChecklist();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Urgent Action Checklist',
            style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        const Text('Immediate Interventions'),
        const SizedBox(height: 4),
        const Text('• Administer Aspirin, Nitroglycerin, Morphine'),
        const Text('• Oxygen Therapy'),
        const Text('• Prepare for PCI'),
        const SizedBox(height: 12),
        const Text('Diagnostic Orders'),
        const SizedBox(height: 4),
        const Text('• Order ECG now / T STAT'),
        const Text('• Order Troponin / T STAT'),
        const Text('• Prepare for chest X-ray'),
      ],
    );
  }
}
