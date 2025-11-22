import 'package:flutter/material.dart';

import '../services/api_client.dart';
import 'analytics_screen.dart';
import 'analytics_screen.dart';
import 'search_screen.dart';
import 'timeline_screen.dart';
import 'clinical_report_screen.dart';
import 'home/dashboard_screen.dart';
import 'home/dashboard_screen.dart';
import 'home/dashboard_screen.dart';
import 'chat/ai_doctor_chat_screen.dart';
import 'agent_timeline_screen.dart';
import 'differential_diagnosis_screen.dart';

class PatientDashboardScreen extends StatefulWidget {
  const PatientDashboardScreen({super.key});

  static const routeName = '/dashboard';

  @override
  State<PatientDashboardScreen> createState() => _PatientDashboardScreenState();
}

class _PatientDashboardScreenState extends State<PatientDashboardScreen> {
  final _api = ApiClient();
  List<Map<String, dynamic>> _cases = const [];
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadCases();
  }

  Future<void> _loadCases({int retries = 10}) async {
    setState(() {
      _loading = true;
      if (retries == 10) _error = null; // Only clear error on first attempt
    });
    try {
      final cases = await _api.listCases();
      if (!mounted) return;
      setState(() {
        _cases = cases;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      
      if (retries > 0) {
        // Wait and retry
        await Future.delayed(const Duration(seconds: 2));
        if (mounted) {
          _loadCases(retries: retries - 1);
        }
      } else {
        setState(() {
          _error = 'Failed to load cases: $e. \nMake sure the backend is running.';
        });
      }
    } finally {
      if (mounted && retries == 0) {
        setState(() {
          _loading = false;
        });
      } else if (mounted && _cases.isNotEmpty) {
         setState(() {
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Patient Overview'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            tooltip: 'New Patient',
            onPressed: () {
              Navigator.pushNamed(context, '/new-patient');
            },
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loading ? null : () => _loadCases(),
          ),
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () {
              // TODO: Navigate to settings screen when implemented
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Today',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Row(
              children: const [
                ChoiceChip(
                  label: Text('All Patients'),
                  selected: true,
                ),
                SizedBox(width: 8),
                ChoiceChip(
                  label: Text('My Cohort'),
                  selected: false,
                ),
              ],
            ),
            const SizedBox(height: 16),
            TextField(
              decoration: InputDecoration(
                prefixIcon: const Icon(Icons.search),
                hintText: 'Search patients...',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
            const SizedBox(height: 16),
            if (_loading) ...[
              const LinearProgressIndicator(),
              const SizedBox(height: 8),
              const Center(child: Text('Connecting to server...', style: TextStyle(color: Colors.grey))),
            ],
            if (_error != null) ...[
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: Colors.red)),
            ],
            const SizedBox(height: 8),
            Expanded(
              child: _cases.isEmpty && !_loading
                  ? const Center(child: Text('No cases yet. Create a new patient to begin.'))
                  : ListView.builder(
                      itemCount: _cases.length,
                      itemBuilder: (context, index) {
                        final c = _cases[index];
                        final id = c['id'] as String? ?? '';
                        final name = c['patient_name'] as String? ?? 'Unknown patient';
                        final status = c['status'] as String? ?? 'Unknown';
                        return Dismissible(
                          key: Key(id),
                          direction: DismissDirection.endToStart,
                          background: Container(
                            alignment: Alignment.centerRight,
                            padding: const EdgeInsets.only(right: 20),
                            color: Colors.red,
                            child: const Icon(Icons.delete, color: Colors.white),
                          ),
                          confirmDismiss: (direction) async {
                            return await showDialog(
                              context: context,
                              builder: (ctx) => AlertDialog(
                                title: const Text('Delete Patient?'),
                                content: Text('Are you sure you want to delete $name? This cannot be undone.'),
                                actions: [
                                  TextButton(
                                    onPressed: () => Navigator.of(ctx).pop(false),
                                    child: const Text('Cancel'),
                                  ),
                                  TextButton(
                                    onPressed: () => Navigator.of(ctx).pop(true),
                                    child: const Text('Delete', style: TextStyle(color: Colors.red)),
                                  ),
                                ],
                              ),
                            );
                          },
                          onDismissed: (direction) async {
                            try {
                              await _api.deleteCase(id);
                              setState(() {
                                _cases.removeAt(index);
                              });
                              if (context.mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(content: Text('$name deleted')),
                                );
                              }
                            } catch (e) {
                              if (context.mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(content: Text('Failed to delete: $e')),
                                );
                                // Refresh list to bring back the item if delete failed
                                _loadCases();
                              }
                            }
                          },
                          child: _PatientCard(
                            name: name,
                            status: status,
                            onOpenTimeline: () {
                              Navigator.pushNamed(
                                context,
                                AgentTimelineScreen.routeName,
                                arguments: id,
                              );
                            },
                            onOpenDiagnosis: () {
                              Navigator.pushNamed(
                                context,
                                DifferentialDiagnosisScreen.routeName,
                                arguments: id,
                              );
                            },
                            onOpenReport: () {
                              Navigator.pushNamed(
                                context,
                                ClinicalReportScreen.routeName,
                                arguments: id,
                              );
                            },
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: 0,
        onDestinationSelected: (index) {
          switch (index) {
            case 0:
              // Already on dashboard
              break;
            case 1:
              Navigator.pushNamed(context, SearchScreen.routeName);
              break;
            case 2:
              Navigator.pushNamed(context, AnalyticsScreen.routeName);
              break;
          }
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.dashboard_outlined),
            selectedIcon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          NavigationDestination(
            icon: Icon(Icons.medication_outlined),
            selectedIcon: Icon(Icons.medication),
            label: 'Medicine',
          ),
          NavigationDestination(
            icon: Icon(Icons.bar_chart_outlined),
            selectedIcon: Icon(Icons.bar_chart),
            label: 'Analytics',
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          Navigator.pushNamed(context, AiDoctorChatScreen.routeName);
        },
        icon: const Icon(Icons.medical_information_outlined),
        label: const Text('AI Doctor'),
      ),
    );
  }
}

class _PatientCard extends StatelessWidget {
  const _PatientCard({
    required this.name,
    required this.status,
    required this.onOpenTimeline,
    required this.onOpenDiagnosis,
    required this.onOpenReport,
  });

  final String name;
  final String status;
  final VoidCallback onOpenTimeline;
  final VoidCallback onOpenDiagnosis;
  final VoidCallback onOpenReport;

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Latest Status: $status',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
            Column(
              children: [
                Row(
                  children: [
                    TextButton(
                      onPressed: onOpenTimeline,
                      child: const Text('Timeline'),
                    ),
                    TextButton(
                      onPressed: onOpenDiagnosis,
                      child: const Text('Diagnosis'),
                    ),
                    TextButton(
                      onPressed: onOpenReport,
                      child: const Text('Report'),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
