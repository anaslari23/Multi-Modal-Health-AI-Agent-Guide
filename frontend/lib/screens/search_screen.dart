import 'package:flutter/material.dart';

import 'analytics_screen.dart';
import 'messages_screen.dart';
import 'patient_dashboard_screen.dart';

class SearchScreen extends StatelessWidget {
  const SearchScreen({super.key});

  static const routeName = '/search';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Search Patients')),
      body: const Center(
        child: Text('Search and filters coming soon'),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 1,
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
              // Already on search
              break;
            case 2:
              Navigator.pushReplacementNamed(context, AnalyticsScreen.routeName);
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
