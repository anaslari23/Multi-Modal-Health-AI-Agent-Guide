import 'package:flutter/material.dart';
import 'dart:async';

import '../services/api_client.dart';
import 'analytics_screen.dart';
import 'analytics_screen.dart';
import 'patient_dashboard_screen.dart';

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  static const routeName = '/search';

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final _searchController = TextEditingController();
  final _api = ApiClient();
  List<Map<String, dynamic>> _results = [];
  bool _loading = false;
  String? _error;
  Timer? _debounce;

  @override
  void initState() {
    super.initState();
    // Load default medicines on startup
    _performSearch('');
  }

  @override
  void dispose() {
    _searchController.dispose();
    _debounce?.cancel();
    super.dispose();
  }

  void _onSearchChanged(String query) {
    if (_debounce?.isActive ?? false) _debounce!.cancel();
    _debounce = Timer(const Duration(milliseconds: 500), () {
      _performSearch(query);
    });
  }

  Future<void> _performSearch(String query) async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final results = await _api.searchMedicines(query);
      if (!mounted) return;
      setState(() {
        _results = results;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Search failed: $e';
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
        title: const Text('Medicine Search'),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: TextField(
              controller: _searchController,
              onChanged: _onSearchChanged,
              decoration: InputDecoration(
                labelText: 'Search Medicines',
                hintText: 'Type to search...',
                prefixIcon: const Icon(Icons.search),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                suffixIcon: _loading
                    ? const Padding(
                        padding: EdgeInsets.all(12),
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchController.clear();
                          _onSearchChanged('');
                        },
                      ),
              ),
            ),
          ),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text(_error!, style: const TextStyle(color: Colors.red)),
            ),
          Expanded(
            child: _results.isEmpty && !_loading
                ? const Center(
                    child: Text(
                      'No medicines found',
                      style: TextStyle(color: Colors.grey),
                    ),
                  )
                : ListView.builder(
                    itemCount: _results.length,
                    itemBuilder: (context, index) {
                      final med = _results[index];
                      return _MedicineTile(medicine: med);
                    },
                  ),
          ),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: 1,
        onDestinationSelected: (index) {
          switch (index) {
            case 0:
              Navigator.pushReplacementNamed(
                  context, PatientDashboardScreen.routeName);
              break;
            case 1:
              // Already on search
              break;
            case 2:
              Navigator.pushReplacementNamed(context, AnalyticsScreen.routeName);
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
    );
  }
}

class _MedicineTile extends StatelessWidget {
  const _MedicineTile({required this.medicine});

  final Map<String, dynamic> medicine;

  @override
  Widget build(BuildContext context) {
    final name = medicine['name'] as String? ?? 'Unknown';
    final price = medicine['price'] as num?;
    final manufacturer = medicine['manufacturer_name'] as String? ?? '';
    final desc = medicine['medicine_desc'] as String? ?? '';
    final sideEffects = medicine['side_effects'] as String? ?? '';
    final composition = medicine['short_composition1'] as String? ?? '';

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: ExpansionTile(
        title: Text(name, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(
          '${price != null ? "₹$price" : "Price N/A"} • $manufacturer',
          style: TextStyle(color: Colors.grey[600]),
        ),
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (composition.isNotEmpty) ...[
                  const Text('Composition:',
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  Text(composition),
                  const SizedBox(height: 8),
                ],
                if (desc.isNotEmpty) ...[
                  const Text('Description:',
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  Text(desc),
                  const SizedBox(height: 8),
                ],
                if (sideEffects.isNotEmpty) ...[
                  const Text('Side Effects:',
                      style: TextStyle(fontWeight: FontWeight.bold, color: Colors.red)),
                  Text(sideEffects),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
