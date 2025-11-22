import 'package:flutter/material.dart';

import 'messages_screen.dart';
import 'patient_dashboard_screen.dart';
import 'search_screen.dart';

class AnalyticsScreen extends StatelessWidget {
  const AnalyticsScreen({super.key});

  static const routeName = '/analytics';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Analytics')),
      body: const Center(
        child: Text('Cohort and risk analytics coming soon'),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 2,
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
              // Already on analytics
              break;
            case 3:
              Navigator.pushReplacementNamed(context, MessagesScreen.routeName);
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
