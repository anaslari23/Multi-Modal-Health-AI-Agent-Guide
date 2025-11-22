import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'core/theme/app_theme.dart';
import 'providers/chat_provider.dart';
import 'providers/case_provider.dart';

import 'screens/onboarding/onboarding_screen.dart';
import 'screens/home/dashboard_screen.dart';
import 'screens/chat/ai_doctor_chat_screen.dart';
import 'screens/intake/symptom_intake_screen.dart';
import 'screens/intake/followup_questions_screen.dart';
import 'screens/intake/data_upload_screen.dart';
import 'screens/intake/vitals_entry_screen.dart';
import 'screens/intake/imaging_upload_screen.dart';
import 'screens/intake/labs_upload_screen.dart';
import 'screens/analysis/diagnosis_result_screen.dart';
import 'screens/analysis/rag_context_screen.dart';
import 'screens/analysis/imaging_review_screen.dart' as analysis_imaging;
import 'screens/analysis/labs_review_screen.dart';
import 'screens/analysis/summary_report_screen.dart';
import 'screens/records/patient_history_screen.dart';
import 'screens/records/case_details_screen.dart';

// Existing screens are still imported for compatibility during refactor.
import 'screens/patient_dashboard_screen.dart';
import 'screens/new_patient_intake_screen.dart';
import 'screens/differential_diagnosis_screen.dart';
import 'screens/imaging_review_screen.dart';
import 'screens/clinical_report_screen.dart';
import 'screens/agent_timeline_screen.dart';
import 'screens/timeline_screen.dart';
import 'screens/search_screen.dart';
import 'screens/analytics_screen.dart';
import 'screens/messages_screen.dart';

void main() {
  runApp(const MMHIEApp());
}

class MMHIEApp extends StatelessWidget {
  const MMHIEApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ChatProvider()),
        ChangeNotifierProvider(create: (_) => CaseProvider()),
      ],
      child: MaterialApp(
        title: 'MM-HIE Agent',
        theme: AppTheme.light(),
        darkTheme: AppTheme.dark(),
        themeMode: ThemeMode.system,
        initialRoute: DashboardScreen.routeName,
        routes: {
          // New structured flows
          OnboardingScreen.routeName: (_) => const OnboardingScreen(),
          DashboardScreen.routeName: (_) => const DashboardScreen(),
          AiDoctorChatScreen.routeName: (_) => const AiDoctorChatScreen(),
          SymptomIntakeScreen.routeName: (_) => const SymptomIntakeScreen(),
          FollowupQuestionsScreen.routeName: (_) => const FollowupQuestionsScreen(),
          DataUploadScreen.routeName: (_) => const DataUploadScreen(),
          VitalsEntryScreen.routeName: (_) => const VitalsEntryScreen(),
          ImagingUploadScreen.routeName: (_) => const ImagingUploadScreen(),
          LabsUploadScreen.routeName: (_) => const LabsUploadScreen(),
          DiagnosisResultScreen.routeName: (_) => const DiagnosisResultScreen(),
          RagContextScreen.routeName: (_) => const RagContextScreen(),
          analysis_imaging.ImagingReviewDetailScreen.routeName: (_) =>
              const analysis_imaging.ImagingReviewDetailScreen(),
          LabsReviewScreen.routeName: (_) => const LabsReviewScreen(),
          SummaryReportScreen.routeName: (_) => const SummaryReportScreen(),
          PatientHistoryScreen.routeName: (_) => const PatientHistoryScreen(),
          CaseDetailsScreen.routeName: (_) => const CaseDetailsScreen(),

          // Legacy screens (kept for now during migration)
          PatientDashboardScreen.routeName: (_) => const PatientDashboardScreen(),
          NewPatientIntakeScreen.routeName: (_) => const NewPatientIntakeScreen(),
          DifferentialDiagnosisScreen.routeName: (_) =>
              const DifferentialDiagnosisScreen(),
          ImagingReviewScreen.routeName: (_) => const ImagingReviewScreen(),
          ClinicalReportScreen.routeName: (_) => const ClinicalReportScreen(),
          AgentTimelineScreen.routeName: (_) => const AgentTimelineScreen(),
          TimelineScreen.routeName: (_) => const TimelineScreen(),
          SearchScreen.routeName: (_) => const SearchScreen(),
          AnalyticsScreen.routeName: (_) => const AnalyticsScreen(),
          MessagesScreen.routeName: (_) => const MessagesScreen(),
        },
      ),
    );
  }
}
