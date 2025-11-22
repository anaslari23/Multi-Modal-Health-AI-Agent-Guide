import 'package:flutter/material.dart';

class PatientHistoryScreen extends StatelessWidget {
  const PatientHistoryScreen({super.key});

  static const routeName = '/records/history';

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(child: Text('Patient History (placeholder)')),
    );
  }
}
