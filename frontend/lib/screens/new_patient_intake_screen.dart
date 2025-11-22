import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

import '../services/api_client.dart';
import '../widgets/vitals_sparkline.dart';

class NewPatientIntakeScreen extends StatefulWidget {
  const NewPatientIntakeScreen({super.key});

  static const routeName = '/new-patient';

  @override
  State<NewPatientIntakeScreen> createState() => _NewPatientIntakeScreenState();
}

class _NewPatientIntakeScreenState extends State<NewPatientIntakeScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  final _nameController = TextEditingController();
  final _dobController = TextEditingController();
  final _patientIdController = TextEditingController();
  final _reasonController = TextEditingController();

  final _symptomTextController = TextEditingController();

  bool _submitting = false;
  String? _error;
  String? _labFilePath;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _nameController.dispose();
    _dobController.dispose();
    _patientIdController.dispose();
    _reasonController.dispose();
    _symptomTextController.dispose();
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _submitToBackend(BuildContext context) async {
    final symptomText = _symptomTextController.text.trim();
    if (symptomText.isEmpty) {
      setState(() {
        _error = 'Please enter symptoms or clinical notes before submitting.';
      });
      return;
    }

    setState(() {
      _submitting = true;
      _error = null;
    });

    final patientId = _patientIdController.text.trim().isEmpty
        ? 'temp-${DateTime.now().millisecondsSinceEpoch}'
        : _patientIdController.text.trim();

    final notes = _reasonController.text.trim();
    final symptoms = symptomText;

    final api = ApiClient();
    try {
      final caseId = await api.createCase(patientId: patientId, notes: notes);
      await api.addSymptoms(caseId: caseId, text: symptoms);

      if (_labFilePath != null) {
        await api.uploadLabReport(caseId: caseId, filePath: _labFilePath!);
      }

      if (!mounted) return;
      Navigator.pushReplacementNamed(
        context,
        '/ddx',
        arguments: caseId,
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to submit to backend: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _submitting = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('New Patient Intake'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'Patient Info'),
            Tab(text: 'Data Upload'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _PatientInfoForm(
            onContinue: () => _tabController.animateTo(1),
          ),
          _DataUploadTab(
            symptomController: _symptomTextController,
            submitting: _submitting,
            error: _error,
            labFilePath: _labFilePath,
            onLabFilePicked: (path) {
              setState(() {
                _labFilePath = path;
              });
              if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Selected lab file: ${path.split('/').last}')),
                );
              }
            },
            onSubmit: () => _submitToBackend(context),
          ),
        ],
      ),
    );
  }
}

class _PatientInfoForm extends StatelessWidget {
  const _PatientInfoForm({required this.onContinue});

  final VoidCallback onContinue;

  @override
  Widget build(BuildContext context) {
    final state = context.findAncestorStateOfType<_NewPatientIntakeScreenState>();
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          TextField(
            decoration: InputDecoration(
              labelText: 'Full Name',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            controller: state?._nameController,
          ),
          const SizedBox(height: 12),
          TextField(
            decoration: InputDecoration(
              labelText: 'Date of Birth (DD/MM/YYYY)',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            readOnly: true,
            onTap: () async {
              final picked = await showDatePicker(
                context: context,
                initialDate: DateTime.now().subtract(const Duration(days: 365 * 30)),
                firstDate: DateTime(1900),
                lastDate: DateTime.now(),
              );
              if (picked != null && state != null) {
                state._dobController.text =
                    '${picked.day.toString().padLeft(2, '0')}/${picked.month.toString().padLeft(2, '0')}/${picked.year}';
              }
            },
            controller: state?._dobController,
          ),
          const SizedBox(height: 12),
          TextField(
            decoration: InputDecoration(
              labelText: 'Patient ID',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            controller: state?._patientIdController,
          ),
          const SizedBox(height: 12),
          TextField(
            decoration: InputDecoration(
              labelText: 'Reason for Visit',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            maxLines: 3,
            controller: state?._reasonController,
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: onContinue,
              child: const Text('Continue to Data Upload'),
            ),
          ),
        ],
      ),
    );
  }
}

class _DataUploadTab extends StatelessWidget {
  const _DataUploadTab({
    required this.symptomController,
    required this.submitting,
    this.error,
    this.labFilePath,
    required this.onLabFilePicked,
    required this.onSubmit,
  });

  final TextEditingController symptomController;
  final bool submitting;
  final String? error;
  final String? labFilePath;
  final Function(String) onLabFilePicked;
  final VoidCallback onSubmit;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _ModalityCard(
            icon: Icons.cloud_upload_outlined,
            title: 'Symptoms',
            subtitle: 'Clinical Notes / Symptoms',
            progress: 0.75,
            actionLabel: 'Upload Text',
            onActionPressed: () {
              // Focus symptom text field or show dialog in future
            },
          ),
          const SizedBox(height: 12),
          _ModalityCard(
            icon: Icons.science_outlined,
            title: 'Structured Labs',
            subtitle: 'Lab Results (HL7/FHIR)',
            progress: 0.3,
            actionLabel: 'Choose Files',
            onActionPressed: () async {
              final result = await FilePicker.platform.pickFiles(
                type: FileType.any,
              );
              if (result != null && result.files.single.path != null) {
                onLabFilePicked(result.files.single.path!);
              }
            },
          ),
          const SizedBox(height: 12),
          _ModalityCard(
            icon: Icons.monitor_heart_outlined,
            title: 'Time-Series Vitals',
            subtitle: 'Monitoring Devices',
            progress: 0.0,
            actionLabel: 'Connect Device',
            onActionPressed: () {},
            footer: const Padding(
              padding: EdgeInsets.only(top: 12),
              child: VitalsSparkline(
                heartRate: [80, 82, 85, 83, 81, 84, 82],
                spo2: [96, 97, 95, 96, 97, 98, 97],
              ),
            ),
          ),
          const SizedBox(height: 24),
          TextField(
            controller: symptomController,
            maxLines: 4,
            decoration: InputDecoration(
              labelText: 'Symptoms / clinical notes (sent to NLP)',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
          const SizedBox(height: 16),
          if (error != null) ...[
            Text(
              error!,
              style: const TextStyle(color: Colors.red),
            ),
            const SizedBox(height: 8),
          ],
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: submitting ? null : onSubmit,
              child: submitting
                  ? const SizedBox(
                      height: 18,
                      width: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Submit for Analysis'),
            ),
          ),
        ],
      ),
    );
  }
}

class _ModalityCard extends StatelessWidget {
  const _ModalityCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.progress,
    required this.actionLabel,
    this.onActionPressed,
    this.footer,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final double progress;
  final String actionLabel;
  final VoidCallback? onActionPressed;
  final Widget? footer;

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            CircleAvatar(
              radius: 24,
              backgroundColor: Theme.of(context).colorScheme.primary.withOpacity(0.1),
              child: Icon(icon, color: Theme.of(context).colorScheme.primary),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 4),
                  Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
                  const SizedBox(height: 8),
                  LinearProgressIndicator(value: progress),
                  if (footer != null) ...[
                    const SizedBox(height: 8),
                    footer!,
                  ],
                ],
              ),
            ),
            const SizedBox(width: 16),
            Column(
              children: [
                Text('${(progress * 100).round()}%'),
                const SizedBox(height: 8),
                OutlinedButton(
                  onPressed: onActionPressed,
                  child: Text(actionLabel),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
