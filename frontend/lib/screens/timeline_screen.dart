import 'package:flutter/material.dart';

import '../services/api_client.dart';

class TimelineScreen extends StatefulWidget {
  const TimelineScreen({super.key});

  static const routeName = '/timeline';

  @override
  State<TimelineScreen> createState() => _TimelineScreenState();
}

class _TimelineScreenState extends State<TimelineScreen> {
  String? _caseId;
  List<Map<String, dynamic>> _events = const [];
  bool _loading = false;
  String? _error;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _caseId ??= ModalRoute.of(context)?.settings.arguments as String?;
    if (_caseId != null && _events.isEmpty && !_loading) {
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
      final data = await api.getTimeline(_caseId!);
      if (!mounted) return;
      setState(() {
        _events = data;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load timeline: $e';
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
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Patient Timeline'),
      ),
      body: Column(
        children: [
          if (_caseId != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Case: $_caseId',
                  style: theme.textTheme.bodySmall,
                ),
              ),
            ),
          if (_loading)
            const LinearProgressIndicator(),
          if (_error != null) ...[
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text(_error!, style: const TextStyle(color: Colors.red)),
            ),
          ],
          const SizedBox(height: 8),
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _events.length,
              itemBuilder: (context, index) {
                final e = _events[index];
                return _TimelineEventTile(event: e);
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _caseId == null
                        ? null
                        : () {
                            Navigator.pushNamed(
                              context,
                              '/ddx',
                              arguments: _caseId,
                            );
                          },
                    child: const Text('Open Differential Diagnosis'),
                  ),
                ),
                const SizedBox(height: 8),
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
                    child: const Text('Open Clinical Report'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _TimelineEventTile extends StatelessWidget {
  const _TimelineEventTile({required this.event});

  final Map<String, dynamic> event;

  @override
  Widget build(BuildContext context) {
    final type = event['type'] as String? ?? 'unknown';
    final createdAt = event['created_at'] as String?;

    final icon = _iconForType(type);
    final title = _titleForType(type);
    final subtitle = _subtitleForType(event);

    final media = _buildMediaPreview(context, event);

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Column(
          children: [
            Icon(icon, size: 24),
            if (event != event) const SizedBox.shrink(),
          ],
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Card(
            margin: const EdgeInsets.only(bottom: 12),
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(title, style: Theme.of(context).textTheme.titleSmall),
                      if (createdAt != null)
                        Text(
                          createdAt,
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: Colors.grey),
                        ),
                    ],
                  ),
                  if (subtitle != null) ...[
                    const SizedBox(height: 4),
                    Text(subtitle,
                        style: Theme.of(context).textTheme.bodySmall),
                  ],
                  if (media != null) ...[
                    const SizedBox(height: 8),
                    media,
                  ],
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  IconData _iconForType(String type) {
    if (type.startsWith('modality_')) {
      final mod = type.split('_').last;
      switch (mod) {
        case 'symptoms':
          return Icons.notes_outlined;
        case 'labs':
          return Icons.science_outlined;
        case 'imaging':
          return Icons.image_outlined;
        case 'vitals':
          return Icons.monitor_heart_outlined;
      }
    }
    switch (type) {
      case 'agent_action':
        return Icons.smart_toy_outlined;
      case 'analysis_result':
        return Icons.analytics_outlined;
      case 'xai_output':
        return Icons.insights_outlined;
      case 'report':
        return Icons.picture_as_pdf_outlined;
      case 'audit_log':
        return Icons.history_outlined;
      default:
        return Icons.circle;
    }
  }

  String _titleForType(String type) {
    if (type.startsWith('modality_')) {
      final mod = type.split('_').last;
      switch (mod) {
        case 'symptoms':
          return 'Symptoms added';
        case 'labs':
          return 'Labs added';
        case 'imaging':
          return 'Imaging added';
        case 'vitals':
          return 'Vitals added';
      }
      return 'Modality update';
    }
    switch (type) {
      case 'agent_action':
        return 'Agent action';
      case 'analysis_result':
        return 'Analysis completed';
      case 'xai_output':
        return 'Explainability outputs';
      case 'report':
        return 'Report generated';
      case 'audit_log':
        return 'Audit log';
      default:
        return type;
    }
  }

  String? _subtitleForType(Map<String, dynamic> event) {
    final type = event['type'] as String? ?? '';
    if (type == 'analysis_result') {
      final triage = event['triage'];
      final risk = event['risk_score'];
      return 'Triage: $triage, risk score: $risk';
    }
    if (type == 'agent_action') {
      final actionType = event['action_type'];
      final cg = event['confidence_gain'];
      return 'Action: $actionType, Δconfidence: ${cg ?? 'n/a'}';
    }
    if (type == 'modality_vitals') {
      final processed = event['processed'] as Map<String, dynamic>?;
      final anomalies = processed?['anomalies'] as List?;
      if (anomalies != null && anomalies.isNotEmpty) {
        return 'Anomalies: ${anomalies.join(', ')}';
      }
    }
    if (type == 'audit_log') {
      final eventName = event['event'];
      final desc = event['description'];
      return '$eventName: $desc';
    }
    return null;
  }

  Widget? _buildMediaPreview(BuildContext context, Map<String, dynamic> event) {
    final type = event['type'] as String? ?? '';
    final api = ApiClient();
    final baseUrl = api.baseUrl;

    if (type == 'xai_output') {
      final labsPath = event['labs_shap_path'] as String?;
      final gradcamPath = event['gradcam_path'] as String?;
      final children = <Widget>[];
      if (labsPath != null && labsPath.isNotEmpty) {
        children.add(
          Expanded(
            child: AspectRatio(
              aspectRatio: 3 / 2,
              child: Image.network(
                '$baseUrl/static/$labsPath',
                fit: BoxFit.cover,
              ),
            ),
          ),
        );
      }
      if (gradcamPath != null && gradcamPath.isNotEmpty) {
        children.add(
          Expanded(
            child: AspectRatio(
              aspectRatio: 3 / 2,
              child: Image.network(
                '$baseUrl/static/$gradcamPath',
                fit: BoxFit.cover,
              ),
            ),
          ),
        );
      }
      if (children.isEmpty) return null;
      return Row(children: children);
    }

    if (type == 'modality_imaging') {
      final processed = event['processed'] as Map<String, dynamic>?;
      final gradcamPath = processed?['gradcam_path'] as String?;
      if (gradcamPath == null || gradcamPath.isEmpty) return null;
      return AspectRatio(
        aspectRatio: 3 / 2,
        child: Image.network(
          '$baseUrl/static/$gradcamPath',
          fit: BoxFit.cover,
        ),
      );
    }

    return null;
  }
}
