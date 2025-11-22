import 'package:flutter/material.dart';

import '../services/api_client.dart';

class ClinicalReportScreen extends StatefulWidget {
  const ClinicalReportScreen({super.key});

  static const routeName = '/report';

  @override
  State<ClinicalReportScreen> createState() => _ClinicalReportScreenState();
}

class _ClinicalReportScreenState extends State<ClinicalReportScreen> {
  String? _caseId;
  Map<String, dynamic>? _analysis;
  Map<String, dynamic>? _report;
  bool _loading = false;
  String? _error;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _caseId ??= ModalRoute.of(context)?.settings.arguments as String?;
    if (_caseId != null && _report == null && !_loading) {
      _loadData();
    }
  }

  Future<void> _loadData() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    final api = ApiClient();
    try {
      if (_caseId == null) return;
      final analysis = await api.getAnalysis(_caseId!);
      final report = await api.getReport(_caseId!);
      if (!mounted) return;
      setState(() {
        _analysis = analysis;
        _report = report;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load report: $e';
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
    final nlp = _analysis?['nlp'] as Map<String, dynamic>?;
    final labs = _analysis?['labs'] as Map<String, dynamic>?;
    final imaging = _analysis?['imaging'] as Map<String, dynamic>?;
    final vitals = _analysis?['vitals'] as Map<String, dynamic>?;
    final xai = _analysis?['xai'] as Map<String, dynamic>?;

    final posterior = _analysis?['posterior_probabilities'] as List<dynamic>?;
    final agentSummary = _analysis?['agent_summary'] as String?;

    final labsShapPath = xai?['labs_shap_path'] as String?;
    final nlpHighlights =
        xai?['nlp_highlights'] as Map<String, dynamic>?;

    // Imaging Grad-CAM URL (served via FastAPI /static mount).
    final gradcamPath = imaging?['gradcam_path'] as String?;
    final gradcamUrl = gradcamPath == null
        ? null
        : 'http://localhost:8000/static/$gradcamPath';

    final riskScore = (fusion?['final_risk_score'] as num?)?.toDouble();
    final triage = fusion?['triage'] as String?;
    final pdfPath = _report?['pdf_path'] as String?;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Clinical Report'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_caseId != null)
              Text('Case: $_caseId',
                  style: Theme.of(context).textTheme.bodySmall),
            Text('Report Preview',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (_loading) const LinearProgressIndicator(),
            if (_error != null) ...[
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: Colors.red)),
            ],
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                padding: const EdgeInsets.all(16),
                child: ListView(
                  children: [
                    const Text('Patient Info'),
                    const SizedBox(height: 4),
                    Text('• ID: ${_caseId ?? '-'}'),
                    const SizedBox(height: 12),
                    const Text('Parsed Labs'),
                    const SizedBox(height: 4),
                    if (labs == null)
                      const Text('• No lab results attached.')
                    else
                      ...[for (final entry in (labs['values'] as Map<String, dynamic>).entries)
                        Text('• ${entry.key}: ${(entry.value['value'] as num).toDouble()} (${entry.value['flag']})')
                      ],
                    const SizedBox(height: 12),
                    const Text('Imaging Findings'),
                    const SizedBox(height: 4),
                    if (imaging == null)
                      const Text('• No imaging attached.')
                    else ...[
                      for (final entry in (imaging['probabilities'] as Map<String, dynamic>).entries)
                        Text('• ${entry.key}: ${(entry.value as num).toDouble().toStringAsFixed(2)}'),
                      if (gradcamUrl != null) ...[
                        const SizedBox(height: 8),
                        const Text('Grad-CAM Overlay'),
                        const SizedBox(height: 4),
                        Image.network(
                          gradcamUrl,
                          height: 200,
                          fit: BoxFit.contain,
                          errorBuilder: (_, __, ___) =>
                              const Text('Failed to load Grad-CAM image'),
                        ),
                      ],
                    ],
                    const SizedBox(height: 12),
                    const Text('NLP Insights'),
                    const SizedBox(height: 4),
                    if (nlp == null)
                      const Text('• No symptom text analysed.')
                    else
                      ...((nlp['conditions'] as List)
                          .cast<Map<String, dynamic>>()
                          .map((c) =>
                              Text('• ${c['condition']}: ${(c['prob'] as num).toDouble().toStringAsFixed(2)}'))),
                    const SizedBox(height: 12),
                    const Text('Vitals Summary'),
                    const SizedBox(height: 4),
                    if (vitals == null)
                      const Text('• No vitals provided.')
                    else ...[
                      Text('• Vitals risk: ${(vitals['vitals_risk'] as num).toDouble().toStringAsFixed(2)}'),
                      Text('• Anomalies: ${(vitals['anomalies'] as List).join(', ')}'),
                    ],
                    const SizedBox(height: 12),
                    const Text('Risk & Triage'),
                    const SizedBox(height: 4),
                    if (riskScore == null || triage == null) ...[
                      const Text('• Risk not yet calculated.'),
                    ] else ...[
                      Text('• Final risk score: ${riskScore.toStringAsFixed(2)}'),
                      Text('• Triage: $triage'),
                    ],
                    const SizedBox(height: 12),
                    const Text('Posterior Differential'),
                    const SizedBox(height: 4),
                    if (posterior == null)
                      const Text('• No posterior probabilities available yet.')
                    else ...[
                      for (final cond in posterior.cast<Map<String, dynamic>>())
                        Text('• ${cond['condition']}: '
                            '${(cond['prob'] as num).toDouble().toStringAsFixed(2)}'),
                    ],
                    const SizedBox(height: 12),
                    const Text('Explanation'),
                    const SizedBox(height: 4),
                    Text(xai?['summary'] as String? ?? 'No explanation generated.'),
                    if (labsShapPath != null) ...[
                      const SizedBox(height: 4),
                      Text('Labs SHAP plot: $labsShapPath'),
                    ],
                    if (nlpHighlights != null && nlpHighlights.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      const Text('Key symptom tokens:'),
                      const SizedBox(height: 4),
                      ...(() {
                        final entries = nlpHighlights.entries
                            .map<MapEntry<String, double>>((e) => MapEntry(
                                e.key,
                                (e.value as num).toDouble()))
                            .toList()
                          ..sort((a, b) => b.value.compareTo(a.value));
                        final top = entries.take(5);
                        return [
                          for (final e in top)
                            Text('• ${e.key}: '
                                '${e.value.toStringAsFixed(2)}'),
                        ];
                      }()),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: (_caseId == null || _report == null)
                    ? null
                    : () {
                        final snack = SnackBar(
                          content: Text('PDF generated at: ${pdfPath ?? 'unknown path'}'),
                        );
                        ScaffoldMessenger.of(context).showSnackBar(snack);
                      },
                icon: const Icon(Icons.picture_as_pdf_outlined),
                label: const Text('Generate & Export PDF'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
