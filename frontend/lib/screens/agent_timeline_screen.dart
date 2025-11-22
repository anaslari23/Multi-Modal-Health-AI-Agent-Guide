import 'package:flutter/material.dart';

import '../services/api_client.dart';

class AgentTimelineScreen extends StatefulWidget {
  const AgentTimelineScreen({super.key});

  static const routeName = '/agent-timeline';

  @override
  State<AgentTimelineScreen> createState() => _AgentTimelineScreenState();
}

class _AgentTimelineScreenState extends State<AgentTimelineScreen> {
  String? _caseId;
  Map<String, dynamic>? _timeline;
  bool _loading = false;
  String? _error;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _caseId ??= ModalRoute.of(context)?.settings.arguments as String?;
    if (_caseId != null && _timeline == null && !_loading) {
      _fetchTimeline();
    }
  }

  Future<void> _fetchTimeline() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = ApiClient();
      final data = await api.getAgentTimeline(_caseId!);
      if (!mounted) return;
      setState(() {
        _timeline = data;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load agent timeline: $e';
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
    final posteriorHistory =
        (_timeline?['posterior_history'] as List?)?.cast<Map<String, dynamic>>() ?? const [];
    final agentActions =
        (_timeline?['agent_actions'] as List?)?.cast<Map<String, dynamic>>() ?? const [];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Agent & Posterior Timeline'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (_caseId != null)
            Text(
              'Case: $_caseId',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          if (_loading) const LinearProgressIndicator(),
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(_error!, style: const TextStyle(color: Colors.red)),
          ],
          const SizedBox(height: 8),
          if (posteriorHistory.isNotEmpty) ...[
            Text('Posterior Updates',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            ...posteriorHistory.map((entry) {
              final step = entry['step'];
              final modalities = entry['modalities'] as Map<String, dynamic>? ?? {};
              final posterior =
                  (entry['posterior'] as List?)?.cast<Map<String, dynamic>>() ?? const [];
              final top = posterior.isNotEmpty ? posterior.first : null;
              final topName = top?['condition'] as String? ?? 'Unknown';
              final topProb = (top?['prob'] as num?)?.toDouble() ?? 0.0;
              return Card(
                child: ListTile(
                  title: Text('Step $step: $topName'),
                  subtitle: Text(
                    'Modalities: '
                    'NLP=${modalities['nlp'] == true ? '✓' : '×'}, '
                    'Labs=${modalities['labs'] == true ? '✓' : '×'}, '
                    'Imaging=${modalities['imaging'] == true ? '✓' : '×'}, '
                    'Vitals=${modalities['vitals'] == true ? '✓' : '×'}\n'
                    'Top posterior: ${(topProb * 100).toStringAsFixed(1)}%',
                  ),
                ),
              );
            }),
            const SizedBox(height: 16),
          ],
          if (agentActions.isNotEmpty) ...[
            Text('Agent Actions',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            ...agentActions.map((a) {
              final trigger = a['trigger'] as String? ?? 'unknown';
              final nextSteps = (a['next_steps'] as List?)?.cast<String>() ?? const [];
              final reasoning = a['agent_reasoning'] as String? ?? '';
              final cg = (a['confidence_gain'] as num?)?.toDouble();
              return Card(
                child: ListTile(
                  title: Text('Trigger: $trigger'),
                  subtitle: Text(
                    [
                      if (cg != null)
                        'Δconfidence: ${cg.toStringAsFixed(2)}',
                      if (nextSteps.isNotEmpty) 'Next: ${nextSteps.first}',
                      if (reasoning.isNotEmpty) 'Reasoning: $reasoning',
                    ].join('\n'),
                  ),
                ),
              );
            }),
          ],
        ],
      ),
    );
  }
}
