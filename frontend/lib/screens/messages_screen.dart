import 'package:flutter/material.dart';

import 'analytics_screen.dart';
import 'patient_dashboard_screen.dart';
import 'search_screen.dart';

class MessagesScreen extends StatelessWidget {
  const MessagesScreen({super.key});

  static const routeName = '/messages';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Messages')),
      body: const Center(
        child: Text('Secure clinician messaging coming soon'),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 3,
        type: BottomNavigationBarType.fixed,
        backgroundColor: Colors.white,
        selectedItemColor: Theme.of(context).colorScheme.primary,
        unselectedItemColor: Colors.grey,
        elevation: 8,
        onTap: (index) {
          switch (index) {
            case 0:
              Navigator.pushReplacementNamed(context, PatientDashboardScreen.routeName);
              break;
            case 1:
              Navigator.pushReplacementNamed(context, SearchScreen.routeName);
              break;
            case 2:
              Navigator.pushReplacementNamed(context, AnalyticsScreen.routeName);
              break;
            case 3:
              // Already on messages
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
    );
  }
}
