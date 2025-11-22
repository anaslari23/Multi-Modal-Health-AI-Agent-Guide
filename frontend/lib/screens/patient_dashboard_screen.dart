import 'package:flutter/material.dart';

import '../services/api_client.dart';
import 'analytics_screen.dart';
import 'messages_screen.dart';
import 'search_screen.dart';
import 'timeline_screen.dart';
import 'clinical_report_screen.dart';
import 'home/dashboard_screen.dart';
import 'chat/ai_doctor_chat_screen.dart';

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

  Future<void> _loadCases() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final cases = await _api.listCases();
      if (!mounted) return;
      setState(() {
        _cases = cases;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load cases: $e';
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
    return Scaffold(
      appBar: AppBar(
        title: const Text('Patient Overview'),
        actions: [
          IconButton(
            icon: const Icon(Icons.medical_information_outlined),
            tooltip: 'AI Doctor Chat',
            onPressed: () {
              Navigator.pushNamed(
                context,
                AiDoctorChatScreen.routeName,
              );
            },
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loading ? null : _loadCases,
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
            if (_loading) const LinearProgressIndicator(),
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
                        return _PatientCard(
                          name: name,
                          status: status,
                          onOpenTimeline: () {
                            Navigator.pushNamed(
                              context,
                              TimelineScreen.routeName,
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
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0,
        type: BottomNavigationBarType.fixed,
        backgroundColor: Colors.white,
        selectedItemColor: Theme.of(context).colorScheme.primary,
        unselectedItemColor: Colors.grey,
        elevation: 8,
        onTap: (index) {
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
            case 3:
              Navigator.pushNamed(context, MessagesScreen.routeName);
              break;
          }
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.dashboard_outlined),
            label: 'Dashboard',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.search),
            label: 'Search',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart_outlined),
            label: 'Analytics',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.message_outlined),
            label: 'Messages',
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          Navigator.pushNamed(context, '/new-patient');
        },
        icon: const Icon(Icons.add),
        label: const Text('New Patient'),
      ),
    );
  }
}

class _PatientCard extends StatelessWidget {
  const _PatientCard({
    required this.name,
    required this.status,
    required this.onOpenTimeline,
    required this.onOpenReport,
  });

  final String name;
  final String status;
  final VoidCallback onOpenTimeline;
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
